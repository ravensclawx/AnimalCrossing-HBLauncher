/*
 * glwgpu.cpp - OpenGL 3.3 subset on WebGPU (Dawn).  See glwgpu.h.
 *
 * Model
 *   - One default framebuffer: an RGBA8 colour + Depth24Plus texture (the
 *     GX EFB).  Draws are recorded into a single render pass that stays open
 *     until something needs the GPU (present, glReadPixels, a full ring).
 *   - Each GL program keeps a CPU copy of its vertex and fragment default
 *     uniform blocks (layout from glslang reflection, std140).  Every draw
 *     appends both blocks to a uniform ring (dynamic offsets); vertex data is
 *     copied into a vertex ring at draw time, as GL would read it then.  The
 *     rings are uploaded with one WriteBuffer each just before a submit.
 *   - GL coordinate conventions are applied in a vertex-shader epilogue:
 *     viewport (may lie outside the framebuffer in GL, not in WebGPU), depth
 *     range and [-1,1] -> [0,1] depth.  A scissor equal to the viewport (and
 *     the GL scissor) restores GL's clipping to the viewport.
 *   - glTexImage2D on a texture already used since the last submit gets a new
 *     GPU texture, so earlier draws keep sampling the old contents.
 */
#include "glwgpu.h"

#include <webgpu/webgpu_cpp.h>
#include <dawn/native/DawnNative.h>
#include <glslang/Public/ShaderLang.h>
#include <glslang/Public/ResourceLimits.h>
#include <glslang/SPIRV/GlslangToSpv.h>

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

typedef unsigned int GLenum;
typedef unsigned int GLuint;
typedef int GLint;
typedef int GLsizei;
typedef unsigned char GLboolean;
typedef unsigned int GLbitfield;
typedef float GLfloat;
typedef double GLdouble;
typedef char GLchar;
typedef unsigned char GLubyte;
typedef intptr_t GLsizeiptr;

/* GL enums used by the port */
enum : GLenum {
    GL_NO_ERROR = 0, GL_POINTS = 0, GL_LINES = 1, GL_LINE_STRIP = 3, GL_TRIANGLES = 4,
    GL_TRIANGLE_STRIP = 5, GL_TRIANGLE_FAN = 6,
    GL_NEVER = 0x200, GL_LESS, GL_EQUAL, GL_LEQUAL, GL_GREATER, GL_NOTEQUAL, GL_GEQUAL, GL_ALWAYS,
    GL_ZERO = 0, GL_ONE = 1, GL_SRC_COLOR = 0x300, GL_ONE_MINUS_SRC_COLOR, GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA, GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA, GL_DST_COLOR, GL_ONE_MINUS_DST_COLOR,
    GL_FRONT = 0x404, GL_BACK = 0x405, GL_FRONT_AND_BACK = 0x408,
    GL_CULL_FACE = 0xB44, GL_DEPTH_TEST = 0xB71, GL_BLEND = 0xBE2, GL_SCISSOR_TEST = 0xC11,
    GL_MULTISAMPLE = 0x809D, GL_TEXTURE_2D = 0xDE1,
    GL_UNSIGNED_BYTE = 0x1401, GL_UNSIGNED_SHORT = 0x1403, GL_UNSIGNED_INT = 0x1405, GL_FLOAT = 0x1406,
    GL_RGBA = 0x1908, GL_VENDOR = 0x1F00, GL_RENDERER, GL_VERSION, GL_EXTENSIONS,
    GL_SHADING_LANGUAGE_VERSION = 0x8B8C,
    GL_NEAREST = 0x2600, GL_LINEAR = 0x2601, GL_NEAREST_MIPMAP_NEAREST = 0x2700,
    GL_LINEAR_MIPMAP_NEAREST, GL_NEAREST_MIPMAP_LINEAR, GL_LINEAR_MIPMAP_LINEAR,
    GL_TEXTURE_MAG_FILTER = 0x2800, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
    GL_REPEAT = 0x2901, GL_CLAMP_TO_EDGE = 0x812F, GL_MIRRORED_REPEAT = 0x8370,
    GL_TEXTURE0 = 0x84C0,
    GL_FUNC_ADD = 0x8006, GL_FUNC_SUBTRACT = 0x800A, GL_FUNC_REVERSE_SUBTRACT = 0x800B,
    GL_MIN = 0x8007, GL_MAX = 0x8008,
    GL_ARRAY_BUFFER = 0x8892, GL_ELEMENT_ARRAY_BUFFER = 0x8893,
    GL_FRAGMENT_SHADER = 0x8B30, GL_VERTEX_SHADER = 0x8B31,
    GL_COMPILE_STATUS = 0x8B81, GL_LINK_STATUS = 0x8B82, GL_INFO_LOG_LENGTH = 0x8B84,
    GL_DEPTH_BUFFER_BIT = 0x100, GL_COLOR_BUFFER_BIT = 0x4000,
    GL_NUM_EXTENSIONS = 0x821D, GL_MAX_TEXTURE_SIZE = 0xD33, GL_MAJOR_VERSION = 0x821B,
    GL_MINOR_VERSION = 0x821C,
};

namespace {

void logf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("[glwgpu] ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

/* ------------------------------------------------------------------ */
/* Objects                                                             */
/* ------------------------------------------------------------------ */

struct Texture {
    wgpu::Texture tex;
    wgpu::TextureView view;
    int w = 0, h = 0;
    GLint minFilter = GL_NEAREST_MIPMAP_LINEAR, magFilter = GL_LINEAR;
    GLint wrapS = GL_REPEAT, wrapT = GL_REPEAT;
    uint64_t usedSerial = 0;              /* batch that last sampled it */
};

struct Buffer {
    std::vector<uint8_t> data;
    wgpu::Buffer gpu;                     /* index buffers: uploaded on use */
    bool gpuDirty = true;
};

struct Attrib {
    bool enabled = false;
    GLint size = 4;
    GLenum type = GL_FLOAT;
    bool normalized = false;
    GLsizei stride = 0;
    uintptr_t offset = 0;
    GLuint buffer = 0;
};

struct VertexArray {
    Attrib attr[8];
    GLuint elementBuffer = 0;
};

struct ShaderObj {
    GLenum type;
    std::string source;
};

struct Uniform {
    std::string name;
    int offset[2] = {-1, -1};             /* in the vertex / fragment block */
    int arrayStride = 0;
    int arraySize = 1;
    int glType = 0;
    int sampler = -1;                     /* index into Program::samplers */
};

struct Program {
    GLuint vs = 0, fs = 0;
    bool linked = false;
    wgpu::ShaderModule module[2];
    std::vector<Uniform> uniforms;
    std::unordered_map<std::string, int> byName;
    std::vector<uint8_t> block[2];
    std::vector<std::string> samplers;    /* sampler uniforms, binding 2+2i / 3+2i */
    std::vector<int> samplerUnit;
    int vpUniform = -1, drUniform = -1;
    wgpu::BindGroupLayout bgl;
    wgpu::PipelineLayout layout;
};

/* GL state that feeds pipelines */
struct PipelineKey {
    GLuint program;
    uint8_t topology, blend, cull, depthTest, depthWrite, colorMask;
    uint16_t blendSrc, blendDst, blendEq, depthFunc;
    bool operator==(const PipelineKey& o) const { return memcmp(this, &o, sizeof *this) == 0; }
};
struct PipelineKeyHash {
    size_t operator()(const PipelineKey& k) const {
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&k);
        size_t h = 1469598103934665603ull;
        for (size_t i = 0; i < sizeof k; i++) h = (h ^ p[i]) * 1099511628211ull;
        return h;
    }
};

/* A CPU-staged ring uploaded once per submit */
struct Ring {
    wgpu::Buffer gpu;
    std::vector<uint8_t> cpu;
    size_t used = 0;
    size_t alloc(size_t bytes, size_t align) {
        size_t at = (used + align - 1) & ~(align - 1);
        if (at + bytes > cpu.size()) return SIZE_MAX;
        used = at + bytes;
        return at;
    }
};

/* ------------------------------------------------------------------ */
/* Context                                                             */
/* ------------------------------------------------------------------ */

struct Context {
    wgpu::Instance instance;
    wgpu::Adapter adapter;
    wgpu::Device device;
    wgpu::Queue queue;
    wgpu::Surface surface;
    wgpu::TextureFormat surfaceFormat = wgpu::TextureFormat::Undefined;
    int surfaceW = 0, surfaceH = 0;

    int fbW = 640, fbH = 480;
    wgpu::Texture color, depth;
    wgpu::TextureView colorView, depthView;

    wgpu::CommandEncoder encoder;
    wgpu::RenderPassEncoder pass;
    bool clearedOnce = false;
    uint64_t batch = 1;

    Ring vertexRing, uniformRing, indexRing;
    uint32_t uniformAlign = 256;

    std::map<GLuint, Texture> textures;
    std::map<GLuint, Buffer> buffers;
    std::map<GLuint, VertexArray> vaos;
    std::map<GLuint, ShaderObj> shaders;
    std::map<GLuint, std::unique_ptr<Program>> programs;
    GLuint nextName = 1;

    /* bindings */
    GLuint boundArray = 0, boundVao = 0, currentProgram = 0;
    GLuint unitTexture[16] = {};
    int activeUnit = 0;

    /* fixed-function state */
    bool blend = false, cull = false, depthTest = false, scissorTest = false;
    GLenum blendSrc = GL_ONE, blendDst = GL_ZERO, blendEq = GL_FUNC_ADD;
    GLenum cullFace = GL_BACK, depthFunc = GL_LESS;
    bool depthMask = true;
    uint8_t colorMask = 0xf;
    GLint vp[4] = {0, 0, 640, 480}, sc[4] = {0, 0, 640, 480};
    float depthNear = 0.0f, depthFar = 1.0f;
    float clearColor[4] = {0, 0, 0, 0};
    float clearDepth = 1.0f;

    std::unordered_map<PipelineKey, wgpu::RenderPipeline, PipelineKeyHash> pipelines;
    std::unordered_map<uint64_t, wgpu::Sampler> samplers;
    std::unordered_map<std::string, wgpu::BindGroup> bindGroups;
    wgpu::Texture whiteTex;
    wgpu::TextureView whiteView;

    wgpu::RenderPipeline clearPipeline[2][16];   /* [depth write][colour mask] */
    wgpu::BindGroupLayout clearBgl;
    wgpu::Buffer clearUniform;
    wgpu::BindGroup clearBg;
    wgpu::RenderPipeline blitPipeline;
    wgpu::BindGroupLayout blitBgl;
    wgpu::Sampler blitSampler;

    std::string dumpDir;
    int dumpEvery = 60;
    uint64_t frame = 0;
};

Context* g;

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

void createFramebuffer() {
    wgpu::TextureDescriptor d{};
    d.size = {static_cast<uint32_t>(g->fbW), static_cast<uint32_t>(g->fbH), 1};
    d.format = wgpu::TextureFormat::RGBA8Unorm;
    d.usage = wgpu::TextureUsage::RenderAttachment | wgpu::TextureUsage::CopySrc |
              wgpu::TextureUsage::TextureBinding;
    g->color = g->device.CreateTexture(&d);
    g->colorView = g->color.CreateView();
    d.format = wgpu::TextureFormat::Depth24Plus;
    d.usage = wgpu::TextureUsage::RenderAttachment;
    g->depth = g->device.CreateTexture(&d);
    g->depthView = g->depth.CreateView();
    g->clearedOnce = false;
}

void makeRing(Ring& r, size_t bytes, wgpu::BufferUsage usage) {
    wgpu::BufferDescriptor d{};
    d.size = bytes;
    d.usage = usage | wgpu::BufferUsage::CopyDst;
    r.gpu = g->device.CreateBuffer(&d);
    r.cpu.resize(bytes);
    r.used = 0;
}

void beginPass() {
    if (g->pass) return;
    if (!g->encoder) g->encoder = g->device.CreateCommandEncoder();
    wgpu::RenderPassColorAttachment ca{};
    ca.view = g->colorView;
    ca.loadOp = g->clearedOnce ? wgpu::LoadOp::Load : wgpu::LoadOp::Clear;
    ca.storeOp = wgpu::StoreOp::Store;
    ca.clearValue = {0, 0, 0, 1};
    wgpu::RenderPassDepthStencilAttachment da{};
    da.view = g->depthView;
    da.depthLoadOp = g->clearedOnce ? wgpu::LoadOp::Load : wgpu::LoadOp::Clear;
    da.depthStoreOp = wgpu::StoreOp::Store;
    da.depthClearValue = 1.0f;
    wgpu::RenderPassDescriptor rp{};
    rp.colorAttachmentCount = 1;
    rp.colorAttachments = &ca;
    rp.depthStencilAttachment = &da;
    g->pass = g->encoder.BeginRenderPass(&rp);
    g->clearedOnce = true;
}

void endPass() {
    if (g->pass) {
        g->pass.End();
        g->pass = nullptr;
    }
}

/* Upload the rings and submit everything recorded so far. */
void submit() {
    endPass();
    for (Ring* r : {&g->vertexRing, &g->uniformRing, &g->indexRing}) {
        if (r->used) g->queue.WriteBuffer(r->gpu, 0, r->cpu.data(), (r->used + 3) & ~size_t(3));
        r->used = 0;
    }
    if (g->encoder) {
        wgpu::CommandBuffer cb = g->encoder.Finish();
        g->queue.Submit(1, &cb);
        g->encoder = nullptr;
    }
    g->batch++;
}

void waitIdle() {
    bool done = false;
    wgpu::Future f = g->queue.OnSubmittedWorkDone(wgpu::CallbackMode::WaitAnyOnly,
                                                  [&](wgpu::QueueWorkDoneStatus, wgpu::StringView) { done = true; });
    g->instance.WaitAny(f, UINT64_MAX);
}

wgpu::CompareFunction compareFn(GLenum f) {
    switch (f) {
    case GL_NEVER: return wgpu::CompareFunction::Never;
    case GL_LESS: return wgpu::CompareFunction::Less;
    case GL_EQUAL: return wgpu::CompareFunction::Equal;
    case GL_LEQUAL: return wgpu::CompareFunction::LessEqual;
    case GL_GREATER: return wgpu::CompareFunction::Greater;
    case GL_NOTEQUAL: return wgpu::CompareFunction::NotEqual;
    case GL_GEQUAL: return wgpu::CompareFunction::GreaterEqual;
    default: return wgpu::CompareFunction::Always;
    }
}

wgpu::BlendFactor blendFactor(GLenum f) {
    switch (f) {
    case GL_ZERO: return wgpu::BlendFactor::Zero;
    case GL_ONE: return wgpu::BlendFactor::One;
    case GL_SRC_COLOR: return wgpu::BlendFactor::Src;
    case GL_ONE_MINUS_SRC_COLOR: return wgpu::BlendFactor::OneMinusSrc;
    case GL_SRC_ALPHA: return wgpu::BlendFactor::SrcAlpha;
    case GL_ONE_MINUS_SRC_ALPHA: return wgpu::BlendFactor::OneMinusSrcAlpha;
    case GL_DST_ALPHA: return wgpu::BlendFactor::DstAlpha;
    case GL_ONE_MINUS_DST_ALPHA: return wgpu::BlendFactor::OneMinusDstAlpha;
    case GL_DST_COLOR: return wgpu::BlendFactor::Dst;
    case GL_ONE_MINUS_DST_COLOR: return wgpu::BlendFactor::OneMinusDst;
    default: return wgpu::BlendFactor::One;
    }
}

wgpu::BlendOperation blendOp(GLenum e) {
    switch (e) {
    case GL_FUNC_SUBTRACT: return wgpu::BlendOperation::Subtract;
    case GL_FUNC_REVERSE_SUBTRACT: return wgpu::BlendOperation::ReverseSubtract;
    case GL_MIN: return wgpu::BlendOperation::Min;
    case GL_MAX: return wgpu::BlendOperation::Max;
    default: return wgpu::BlendOperation::Add;
    }
}

wgpu::AddressMode addressMode(GLint w) {
    switch (w) {
    case GL_CLAMP_TO_EDGE: return wgpu::AddressMode::ClampToEdge;
    case GL_MIRRORED_REPEAT: return wgpu::AddressMode::MirrorRepeat;
    default: return wgpu::AddressMode::Repeat;
    }
}

wgpu::Sampler samplerFor(const Texture& t) {
    uint64_t key = (uint64_t(t.minFilter) << 48) ^ (uint64_t(t.magFilter) << 32) ^
                   (uint64_t(t.wrapS) << 16) ^ uint64_t(t.wrapT);
    auto it = g->samplers.find(key);
    if (it != g->samplers.end()) return it->second;
    wgpu::SamplerDescriptor d{};
    d.addressModeU = addressMode(t.wrapS);
    d.addressModeV = addressMode(t.wrapT);
    d.magFilter = t.magFilter == GL_NEAREST ? wgpu::FilterMode::Nearest : wgpu::FilterMode::Linear;
    bool minLinear = t.minFilter == GL_LINEAR || t.minFilter == GL_LINEAR_MIPMAP_NEAREST ||
                     t.minFilter == GL_LINEAR_MIPMAP_LINEAR;
    d.minFilter = minLinear ? wgpu::FilterMode::Linear : wgpu::FilterMode::Nearest;
    wgpu::Sampler s = g->device.CreateSampler(&d);
    g->samplers[key] = s;
    return s;
}

/* ------------------------------------------------------------------ */
/* Shaders                                                             */
/* ------------------------------------------------------------------ */

const char* kVertexEpilogue = R"(
uniform vec4 glwgpu_viewport;   /* xy scale, zw offset (NDC) */
uniform vec2 glwgpu_depthrange; /* near, far */
void main() {
    glwgpu_user_main();
    vec4 p = gl_Position;
    p.xy = p.xy * glwgpu_viewport.xy + glwgpu_viewport.zw * p.w;
    float z01 = (p.z + p.w) * 0.5;
    p.z = glwgpu_depthrange.x * p.w + (glwgpu_depthrange.y - glwgpu_depthrange.x) * z01;
    gl_Position = p;
}
)";

/* Rewrite GL-flavoured GLSL for the Vulkan rules: combined samplers become a
 * texture + sampler pair (bindings 2+2i, 3+2i) behind a macro so texture()
 * calls compile unchanged; the vertex main gets the epilogue. */
std::string rewriteSource(const std::string& src, GLenum type, std::vector<std::string>& samplers) {
    std::string out;
    static const std::regex samplerDecl(R"(^\s*uniform\s+sampler2D\s+(\w+)\s*;)");
    size_t pos = 0;
    while (pos < src.size()) {
        size_t eol = src.find('\n', pos);
        if (eol == std::string::npos) eol = src.size();
        std::string line = src.substr(pos, eol - pos);
        std::smatch m;
        if (std::regex_search(line, m, samplerDecl)) {
            std::string name = m[1];
            int index = -1;
            for (size_t i = 0; i < samplers.size(); i++)
                if (samplers[i] == name) index = static_cast<int>(i);
            if (index < 0) {
                index = static_cast<int>(samplers.size());
                samplers.push_back(name);
            }
            char buf[512];
            snprintf(buf, sizeof buf,
                     "layout(set = 0, binding = %d) uniform texture2D %s_glwgpu_t;\n"
                     "layout(set = 0, binding = %d) uniform sampler %s_glwgpu_s;\n"
                     "#define %s sampler2D(%s_glwgpu_t, %s_glwgpu_s)",
                     2 + 2 * index, name.c_str(), 3 + 2 * index, name.c_str(), name.c_str(), name.c_str(),
                     name.c_str());
            out += buf;
        } else {
            out += line;
        }
        out += '\n';
        pos = eol + 1;
    }
    /* explicit bindings need 4.20+; nothing else the port uses changes */
    static const std::regex version(R"(#version\s+\d+(\s+core)?)");
    out = std::regex_replace(out, version, "#version 450 core", std::regex_constants::format_first_only);
    if (type == GL_VERTEX_SHADER) {
        static const std::regex mainDecl(R"(void\s+main\s*\(\s*(void)?\s*\))");
        out = std::regex_replace(out, mainDecl, "void glwgpu_user_main()", std::regex_constants::format_first_only);
        out += kVertexEpilogue;
    }
    return out;
}

struct StageResult {
    std::vector<uint32_t> spirv;
    struct Var { std::string name; int offset, arrayStride, arraySize, glType; };
    std::vector<Var> vars;
    std::string log;
};

bool compileStage(const std::string& src, GLenum type, int uboBinding, StageResult& out) {
    EShLanguage lang = type == GL_VERTEX_SHADER ? EShLangVertex : EShLangFragment;
    glslang::TShader sh(lang);
    const char* s = src.c_str();
    sh.setStrings(&s, 1);
    sh.setEnvInput(glslang::EShSourceGlsl, lang, glslang::EShClientVulkan, 100);
    sh.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
    sh.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);
    sh.setEnvInputVulkanRulesRelaxed();
    sh.setAutoMapLocations(true);
    sh.setAutoMapBindings(true);
    sh.setGlobalUniformBlockName(type == GL_VERTEX_SHADER ? "glwgpu_vs_uniforms" : "glwgpu_fs_uniforms");
    sh.setGlobalUniformSet(0);
    sh.setGlobalUniformBinding(uboBinding);
    EShMessages msgs = static_cast<EShMessages>(EShMsgSpvRules | EShMsgVulkanRules);
    if (!sh.parse(GetDefaultResources(), 330, ECoreProfile, false, false, msgs)) {
        out.log = sh.getInfoLog();
        return false;
    }
    glslang::TProgram prog;
    prog.addShader(&sh);
    if (!prog.link(msgs) || !prog.mapIO()) {
        out.log = prog.getInfoLog();
        return false;
    }
    prog.buildReflection(EShReflectionAllBlockVariables);
    for (int i = 0; i < prog.getNumUniformVariables(); i++) {
        const glslang::TObjectReflection& u = prog.getUniform(i);
        if (u.offset < 0) continue;                      /* opaque (texture/sampler) */
        StageResult::Var v{u.name, u.offset, u.arrayStride, u.size > 0 ? u.size : 1, u.glDefineType};
        out.vars.push_back(v);
    }
    glslang::GlslangToSpv(*prog.getIntermediate(lang), out.spirv);
    return true;
}

wgpu::ShaderModule makeModule(const std::vector<uint32_t>& spirv) {
    wgpu::ShaderSourceSPIRV spv{};
    spv.codeSize = static_cast<uint32_t>(spirv.size());
    spv.code = spirv.data();
    wgpu::ShaderModuleDescriptor d{};
    d.nextInChain = &spv;
    return g->device.CreateShaderModule(&d);
}

bool linkProgram(Program& p) {
    auto vsIt = g->shaders.find(p.vs), fsIt = g->shaders.find(p.fs);
    if (vsIt == g->shaders.end() || fsIt == g->shaders.end()) return false;
    p.samplers.clear();
    std::string vsrc = rewriteSource(vsIt->second.source, GL_VERTEX_SHADER, p.samplers);
    std::string fsrc = rewriteSource(fsIt->second.source, GL_FRAGMENT_SHADER, p.samplers);
    StageResult r[2];
    if (!compileStage(vsrc, GL_VERTEX_SHADER, 0, r[0])) {
        logf("vertex shader failed:\n%s", r[0].log.c_str());
        return false;
    }
    if (!compileStage(fsrc, GL_FRAGMENT_SHADER, 1, r[1])) {
        logf("fragment shader failed:\n%s", r[1].log.c_str());
        return false;
    }
    p.uniforms.clear();
    p.byName.clear();
    for (int stage = 0; stage < 2; stage++) {
        int size = 16;
        for (const auto& v : r[stage].vars) {
            auto it = p.byName.find(v.name);
            int index;
            if (it == p.byName.end()) {
                index = static_cast<int>(p.uniforms.size());
                p.uniforms.push_back({});
                p.uniforms[index].name = v.name;
                p.byName[v.name] = index;
            } else {
                index = it->second;
            }
            Uniform& u = p.uniforms[index];
            u.offset[stage] = v.offset;
            u.arrayStride = v.arrayStride;
            u.arraySize = v.arraySize;
            u.glType = v.glType;
            int end = v.offset + (v.arraySize > 1 ? v.arraySize * v.arrayStride : 64);
            if (end > size) size = end;
        }
        p.block[stage].assign((size + 255) & ~255, 0);
    }
    for (size_t i = 0; i < p.samplers.size(); i++) {
        Uniform u;
        u.name = p.samplers[i];
        u.sampler = static_cast<int>(i);
        p.byName[u.name] = static_cast<int>(p.uniforms.size());
        p.uniforms.push_back(u);
    }
    p.samplerUnit.assign(p.samplers.size(), 0);
    p.vpUniform = p.byName.count("glwgpu_viewport") ? p.byName["glwgpu_viewport"] : -1;
    p.drUniform = p.byName.count("glwgpu_depthrange") ? p.byName["glwgpu_depthrange"] : -1;
    p.module[0] = makeModule(r[0].spirv);
    p.module[1] = makeModule(r[1].spirv);

    std::vector<wgpu::BindGroupLayoutEntry> e(2 + 2 * p.samplers.size());
    e[0].binding = 0;
    e[0].visibility = wgpu::ShaderStage::Vertex;
    e[0].buffer.type = wgpu::BufferBindingType::Uniform;
    e[0].buffer.hasDynamicOffset = true;
    e[0].buffer.minBindingSize = p.block[0].size();
    e[1].binding = 1;
    e[1].visibility = wgpu::ShaderStage::Fragment;
    e[1].buffer.type = wgpu::BufferBindingType::Uniform;
    e[1].buffer.hasDynamicOffset = true;
    e[1].buffer.minBindingSize = p.block[1].size();
    for (size_t i = 0; i < p.samplers.size(); i++) {
        auto& t = e[2 + 2 * i];
        t.binding = static_cast<uint32_t>(2 + 2 * i);
        t.visibility = wgpu::ShaderStage::Fragment | wgpu::ShaderStage::Vertex;
        t.texture.sampleType = wgpu::TextureSampleType::Float;
        t.texture.viewDimension = wgpu::TextureViewDimension::e2D;
        auto& s = e[3 + 2 * i];
        s.binding = static_cast<uint32_t>(3 + 2 * i);
        s.visibility = wgpu::ShaderStage::Fragment | wgpu::ShaderStage::Vertex;
        s.sampler.type = wgpu::SamplerBindingType::Filtering;
    }
    wgpu::BindGroupLayoutDescriptor bd{};
    bd.entryCount = e.size();
    bd.entries = e.data();
    p.bgl = g->device.CreateBindGroupLayout(&bd);
    wgpu::PipelineLayoutDescriptor pd{};
    pd.bindGroupLayoutCount = 1;
    pd.bindGroupLayouts = &p.bgl;
    p.layout = g->device.CreatePipelineLayout(&pd);
    p.linked = true;
    return true;
}

/* ------------------------------------------------------------------ */
/* Draws                                                               */
/* ------------------------------------------------------------------ */

wgpu::VertexFormat vertexFormat(const Attrib& a) {
    if (a.type == GL_FLOAT) {
        switch (a.size) {
        case 1: return wgpu::VertexFormat::Float32;
        case 2: return wgpu::VertexFormat::Float32x2;
        case 3: return wgpu::VertexFormat::Float32x3;
        default: return wgpu::VertexFormat::Float32x4;
        }
    }
    if (a.type == GL_UNSIGNED_BYTE)
        return a.normalized ? wgpu::VertexFormat::Unorm8x4 : wgpu::VertexFormat::Uint8x4;
    return wgpu::VertexFormat::Float32x4;
}

wgpu::RenderPipeline pipelineFor(Program& p, const PipelineKey& key, const VertexArray& vao) {
    auto it = g->pipelines.find(key);
    if (it != g->pipelines.end()) return it->second;

    std::vector<wgpu::VertexAttribute> attrs;
    uint64_t stride = 0;
    for (int i = 0; i < 8; i++) {
        const Attrib& a = vao.attr[i];
        if (!a.enabled) continue;
        wgpu::VertexAttribute va{};
        va.format = vertexFormat(a);
        va.offset = a.offset;
        va.shaderLocation = i;
        attrs.push_back(va);
        stride = a.stride;
    }
    wgpu::VertexBufferLayout vbl{};
    vbl.arrayStride = stride;
    vbl.attributeCount = attrs.size();
    vbl.attributes = attrs.data();

    wgpu::BlendState bs{};
    bs.color.srcFactor = blendFactor(key.blendSrc);
    bs.color.dstFactor = blendFactor(key.blendDst);
    bs.color.operation = blendOp(key.blendEq);
    bs.alpha = bs.color;
    wgpu::ColorTargetState ct{};
    ct.format = wgpu::TextureFormat::RGBA8Unorm;
    ct.blend = key.blend ? &bs : nullptr;
    ct.writeMask = static_cast<wgpu::ColorWriteMask>(key.colorMask);
    wgpu::FragmentState fs{};
    fs.module = p.module[1];
    fs.entryPoint = "main";
    fs.targetCount = 1;
    fs.targets = &ct;

    wgpu::DepthStencilState ds{};
    ds.format = wgpu::TextureFormat::Depth24Plus;
    ds.depthWriteEnabled = key.depthTest && key.depthWrite ? wgpu::OptionalBool::True : wgpu::OptionalBool::False;
    ds.depthCompare = key.depthTest ? compareFn(key.depthFunc) : wgpu::CompareFunction::Always;

    wgpu::RenderPipelineDescriptor d{};
    d.layout = p.layout;
    d.vertex.module = p.module[0];
    d.vertex.entryPoint = "main";
    d.vertex.bufferCount = 1;
    d.vertex.buffers = &vbl;
    d.primitive.topology = static_cast<wgpu::PrimitiveTopology>(key.topology);
    d.primitive.frontFace = wgpu::FrontFace::CCW;
    d.primitive.cullMode = key.cull == 1   ? wgpu::CullMode::Front
                           : key.cull == 2 ? wgpu::CullMode::Back
                                           : wgpu::CullMode::None;
    d.depthStencil = &ds;
    d.fragment = &fs;
    wgpu::RenderPipeline rp = g->device.CreateRenderPipeline(&d);
    g->pipelines[key] = rp;
    return rp;
}

Texture* textureForUnit(int unit) {
    GLuint id = g->unitTexture[unit];
    auto it = g->textures.find(id);
    if (it == g->textures.end() || !it->second.tex) return nullptr;
    return &it->second;
}

/* Viewport (GL: bottom-left origin) as NDC scale/offset into the full
 * framebuffer, plus the scissor that clips to it.  Returns false if empty. */
bool viewportTransform(float out[4], uint32_t rect[4]) {
    float W = static_cast<float>(g->fbW), H = static_cast<float>(g->fbH);
    float vx = static_cast<float>(g->vp[0]), vw = static_cast<float>(g->vp[2]), vh = static_cast<float>(g->vp[3]);
    float vyTop = H - static_cast<float>(g->vp[1] + g->vp[3]);
    out[0] = vw / W;
    out[1] = vh / H;
    out[2] = (2.0f * vx + vw) / W - 1.0f;
    out[3] = 1.0f - (2.0f * vyTop + vh) / H;
    int x0 = g->vp[0], y0 = g->fbH - (g->vp[1] + g->vp[3]);
    int x1 = x0 + g->vp[2], y1 = y0 + g->vp[3];
    if (g->scissorTest) {
        int sx0 = g->sc[0], sy0 = g->fbH - (g->sc[1] + g->sc[3]);
        int sx1 = sx0 + g->sc[2], sy1 = sy0 + g->sc[3];
        if (sx0 > x0) x0 = sx0;
        if (sy0 > y0) y0 = sy0;
        if (sx1 < x1) x1 = sx1;
        if (sy1 < y1) y1 = sy1;
    }
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > g->fbW) x1 = g->fbW;
    if (y1 > g->fbH) y1 = g->fbH;
    if (x1 <= x0 || y1 <= y0) return false;
    rect[0] = x0; rect[1] = y0; rect[2] = x1 - x0; rect[3] = y1 - y0;
    return true;
}

void setUniformRaw(Program& p, int index, int element, const void* data, size_t bytes);

void draw(GLenum mode, GLint first, GLsizei count, GLenum indexType, uintptr_t indexOffset, bool indexed) {
    if (count <= 0 || !g->currentProgram) return;
    auto pit = g->programs.find(g->currentProgram);
    if (pit == g->programs.end() || !pit->second->linked) return;
    Program& p = *pit->second;
    if (g->cull && g->cullFace == GL_FRONT_AND_BACK &&
        (mode == GL_TRIANGLES || mode == GL_TRIANGLE_STRIP || mode == GL_TRIANGLE_FAN))
        return;
    VertexArray& vao = g->vaos[g->boundVao];

    float vpt[4];
    uint32_t rect[4];
    if (!viewportTransform(vpt, rect)) return;
    if (p.vpUniform >= 0) setUniformRaw(p, p.vpUniform, 0, vpt, sizeof vpt);
    if (p.drUniform >= 0) {
        float dr[2] = {g->depthNear, g->depthFar};
        setUniformRaw(p, p.drUniform, 0, dr, sizeof dr);
    }

    /* Vertex data: the whole bound array buffer, as GL reads it now */
    GLuint vb = 0;
    for (auto& a : vao.attr)
        if (a.enabled) vb = a.buffer;
    Buffer& vbuf = g->buffers[vb];
    size_t vbytes = vbuf.data.size();
    if (!indexed) {
        int stride = 0;
        for (auto& a : vao.attr)
            if (a.enabled) stride = a.stride;
        vbytes = std::min(vbytes, static_cast<size_t>(first + count) * stride);
    }

    /* Index data: triangle fans become lists */
    std::vector<uint32_t> fan;
    wgpu::PrimitiveTopology topo;
    switch (mode) {
    case GL_POINTS: topo = wgpu::PrimitiveTopology::PointList; break;
    case GL_LINES: topo = wgpu::PrimitiveTopology::LineList; break;
    case GL_LINE_STRIP: topo = wgpu::PrimitiveTopology::LineStrip; break;
    case GL_TRIANGLE_STRIP: topo = wgpu::PrimitiveTopology::TriangleStrip; break;
    default: topo = wgpu::PrimitiveTopology::TriangleList; break;
    }
    if (mode == GL_TRIANGLE_FAN) {
        if (indexed || count < 3) return;
        for (int i = 1; i + 1 < count; i++) {
            fan.push_back(first);
            fan.push_back(first + i);
            fan.push_back(first + i + 1);
        }
    }

    size_t uboBytes = p.block[0].size() + p.block[1].size() + 2 * g->uniformAlign;
    if (g->vertexRing.used + vbytes + 16 > g->vertexRing.cpu.size() ||
        g->uniformRing.used + uboBytes > g->uniformRing.cpu.size() ||
        g->indexRing.used + fan.size() * 4 + 16 > g->indexRing.cpu.size())
        submit();

    beginPass();
    size_t voff = g->vertexRing.alloc(vbytes ? vbytes : 4, 16);
    if (vbytes) memcpy(&g->vertexRing.cpu[voff], vbuf.data.data(), vbytes);
    size_t u0 = g->uniformRing.alloc(p.block[0].size(), g->uniformAlign);
    memcpy(&g->uniformRing.cpu[u0], p.block[0].data(), p.block[0].size());
    size_t u1 = g->uniformRing.alloc(p.block[1].size(), g->uniformAlign);
    memcpy(&g->uniformRing.cpu[u1], p.block[1].data(), p.block[1].size());

    /* Bind group: uniform rings + the textures on the program's units */
    std::string bgKey = std::to_string(g->currentProgram);
    std::vector<wgpu::BindGroupEntry> be(2 + 2 * p.samplers.size());
    be[0].binding = 0;
    be[0].buffer = g->uniformRing.gpu;
    be[0].size = p.block[0].size();
    be[1].binding = 1;
    be[1].buffer = g->uniformRing.gpu;
    be[1].size = p.block[1].size();
    for (size_t i = 0; i < p.samplers.size(); i++) {
        Texture* t = textureForUnit(p.samplerUnit[i]);
        wgpu::TextureView view = t ? t->view : g->whiteView;
        wgpu::Sampler smp = t ? samplerFor(*t) : samplerFor(Texture{});
        if (t) t->usedSerial = g->batch;
        be[2 + 2 * i].binding = static_cast<uint32_t>(2 + 2 * i);
        be[2 + 2 * i].textureView = view;
        be[3 + 2 * i].binding = static_cast<uint32_t>(3 + 2 * i);
        be[3 + 2 * i].sampler = smp;
        char k[64];
        snprintf(k, sizeof k, ":%p/%p", static_cast<void*>(view.Get()), static_cast<void*>(smp.Get()));
        bgKey += k;
    }
    wgpu::BindGroup bg;
    auto bit = g->bindGroups.find(bgKey);
    if (bit != g->bindGroups.end()) {
        bg = bit->second;
    } else {
        wgpu::BindGroupDescriptor bd{};
        bd.layout = p.bgl;
        bd.entryCount = be.size();
        bd.entries = be.data();
        bg = g->device.CreateBindGroup(&bd);
        if (g->bindGroups.size() > 4096) g->bindGroups.clear();
        g->bindGroups[bgKey] = bg;
    }

    PipelineKey key{};
    key.program = g->currentProgram;
    key.topology = static_cast<uint8_t>(topo);
    key.blend = g->blend;
    key.blendSrc = static_cast<uint16_t>(g->blendSrc);
    key.blendDst = static_cast<uint16_t>(g->blendDst);
    key.blendEq = static_cast<uint16_t>(g->blendEq);
    key.cull = !g->cull ? 0 : g->cullFace == GL_FRONT ? 1 : 2;
    key.depthTest = g->depthTest;
    key.depthWrite = g->depthMask;
    key.depthFunc = static_cast<uint16_t>(g->depthFunc);
    key.colorMask = g->colorMask;
    wgpu::RenderPipeline pipe = pipelineFor(p, key, vao);

    uint32_t dyn[2] = {static_cast<uint32_t>(u0), static_cast<uint32_t>(u1)};
    g->pass.SetPipeline(pipe);
    g->pass.SetBindGroup(0, bg, 2, dyn);
    g->pass.SetScissorRect(rect[0], rect[1], rect[2], rect[3]);
    g->pass.SetVertexBuffer(0, g->vertexRing.gpu, voff, vbytes ? vbytes : 4);
    if (!fan.empty()) {
        size_t ioff = g->indexRing.alloc(fan.size() * 4, 16);
        memcpy(&g->indexRing.cpu[ioff], fan.data(), fan.size() * 4);
        g->pass.SetIndexBuffer(g->indexRing.gpu, wgpu::IndexFormat::Uint32, ioff, fan.size() * 4);
        g->pass.DrawIndexed(static_cast<uint32_t>(fan.size()));
    } else if (indexed) {
        Buffer& eb = g->buffers[vao.elementBuffer];
        if (eb.gpuDirty || !eb.gpu) {
            wgpu::BufferDescriptor d{};
            d.size = (eb.data.size() + 3) & ~size_t(3);
            d.usage = wgpu::BufferUsage::Index | wgpu::BufferUsage::CopyDst;
            eb.gpu = g->device.CreateBuffer(&d);
            std::vector<uint8_t> padded(eb.data);
            padded.resize(d.size);
            g->queue.WriteBuffer(eb.gpu, 0, padded.data(), padded.size());
            eb.gpuDirty = false;
        }
        bool u16 = indexType == GL_UNSIGNED_SHORT;
        g->pass.SetIndexBuffer(eb.gpu, u16 ? wgpu::IndexFormat::Uint16 : wgpu::IndexFormat::Uint32, 0,
                               eb.gpu.GetSize());
        g->pass.DrawIndexed(count, 1, static_cast<uint32_t>(indexOffset / (u16 ? 2 : 4)));
    } else {
        g->pass.Draw(count, 1, first);
    }
}

/* Fill the viewport-independent clear rectangle with a fullscreen triangle. */
void clear(GLbitfield mask) {
    bool doColor = (mask & GL_COLOR_BUFFER_BIT) && g->colorMask;
    bool doDepth = (mask & GL_DEPTH_BUFFER_BIT) && g->depthMask;
    if (!doColor && !doDepth) return;
    uint32_t x0 = 0, y0 = 0, x1 = g->fbW, y1 = g->fbH;
    if (g->scissorTest) {
        int sx0 = g->sc[0], sy0 = g->fbH - (g->sc[1] + g->sc[3]);
        x0 = std::max(0, sx0);
        y0 = std::max(0, sy0);
        x1 = std::min(g->fbW, sx0 + g->sc[2]);
        y1 = std::min(g->fbH, sy0 + g->sc[3]);
        if (x1 <= x0 || y1 <= y0) return;
    }
    /* Whole-target clear with full masks: restart the pass with a clear op */
    if (x0 == 0 && y0 == 0 && x1 == static_cast<uint32_t>(g->fbW) && y1 == static_cast<uint32_t>(g->fbH) &&
        doColor && doDepth && g->colorMask == 0xf) {
        endPass();
        if (!g->encoder) g->encoder = g->device.CreateCommandEncoder();
        wgpu::RenderPassColorAttachment ca{};
        ca.view = g->colorView;
        ca.loadOp = wgpu::LoadOp::Clear;
        ca.storeOp = wgpu::StoreOp::Store;
        ca.clearValue = {g->clearColor[0], g->clearColor[1], g->clearColor[2], g->clearColor[3]};
        wgpu::RenderPassDepthStencilAttachment da{};
        da.view = g->depthView;
        da.depthLoadOp = wgpu::LoadOp::Clear;
        da.depthStoreOp = wgpu::StoreOp::Store;
        da.depthClearValue = g->clearDepth;
        wgpu::RenderPassDescriptor rp{};
        rp.colorAttachmentCount = 1;
        rp.colorAttachments = &ca;
        rp.depthStencilAttachment = &da;
        g->pass = g->encoder.BeginRenderPass(&rp);
        g->clearedOnce = true;
        return;
    }
    beginPass();
    float u[8] = {g->clearColor[0], g->clearColor[1], g->clearColor[2], g->clearColor[3], g->clearDepth, 0, 0, 0};
    g->queue.WriteBuffer(g->clearUniform, 0, u, sizeof u);   /* fine: clears are rare */
    int maskIndex = doColor ? g->colorMask : 0;
    g->pass.SetPipeline(g->clearPipeline[doDepth ? 1 : 0][maskIndex]);
    g->pass.SetBindGroup(0, g->clearBg);
    g->pass.SetScissorRect(x0, y0, x1 - x0, y1 - y0);
    g->pass.Draw(3);
    /* the uniform write lands before this submit; force one so a second
     * clear with other values in the same batch is not overwritten */
    submit();
}

void createHelpers() {
    const char* clearWgsl = R"(
struct U { color : vec4f, depth : vec4f };
@group(0) @binding(0) var<uniform> u : U;
@vertex fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {
  let p = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(p[i], u.depth.x, 1.0);
}
@fragment fn fs() -> @location(0) vec4f { return u.color; }
)";
    wgpu::ShaderSourceWGSL w{};
    w.code = clearWgsl;
    wgpu::ShaderModuleDescriptor md{};
    md.nextInChain = &w;
    wgpu::ShaderModule m = g->device.CreateShaderModule(&md);
    wgpu::BindGroupLayoutEntry e{};
    e.binding = 0;
    e.visibility = wgpu::ShaderStage::Vertex | wgpu::ShaderStage::Fragment;
    e.buffer.type = wgpu::BufferBindingType::Uniform;
    wgpu::BindGroupLayoutDescriptor bd{};
    bd.entryCount = 1;
    bd.entries = &e;
    g->clearBgl = g->device.CreateBindGroupLayout(&bd);
    wgpu::PipelineLayoutDescriptor pd{};
    pd.bindGroupLayoutCount = 1;
    pd.bindGroupLayouts = &g->clearBgl;
    wgpu::PipelineLayout pl = g->device.CreatePipelineLayout(&pd);
    wgpu::BufferDescriptor ub{};
    ub.size = 32;
    ub.usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst;
    g->clearUniform = g->device.CreateBuffer(&ub);
    wgpu::BindGroupEntry be{};
    be.binding = 0;
    be.buffer = g->clearUniform;
    be.size = 32;
    wgpu::BindGroupDescriptor bgd{};
    bgd.layout = g->clearBgl;
    bgd.entryCount = 1;
    bgd.entries = &be;
    g->clearBg = g->device.CreateBindGroup(&bgd);
    for (int depth = 0; depth < 2; depth++) {
        for (int mask = 0; mask < 16; mask++) {
            wgpu::ColorTargetState ct{};
            ct.format = wgpu::TextureFormat::RGBA8Unorm;
            ct.writeMask = static_cast<wgpu::ColorWriteMask>(mask);
            wgpu::FragmentState fs{};
            fs.module = m;
            fs.entryPoint = "fs";
            fs.targetCount = 1;
            fs.targets = &ct;
            wgpu::DepthStencilState ds{};
            ds.format = wgpu::TextureFormat::Depth24Plus;
            ds.depthWriteEnabled = depth ? wgpu::OptionalBool::True : wgpu::OptionalBool::False;
            ds.depthCompare = wgpu::CompareFunction::Always;
            wgpu::RenderPipelineDescriptor d{};
            d.layout = pl;
            d.vertex.module = m;
            d.vertex.entryPoint = "vs";
            d.depthStencil = &ds;
            d.fragment = &fs;
            g->clearPipeline[depth][mask] = g->device.CreateRenderPipeline(&d);
        }
    }

    /* 1x1 white for units with nothing bound */
    wgpu::TextureDescriptor td{};
    td.size = {1, 1, 1};
    td.format = wgpu::TextureFormat::RGBA8Unorm;
    td.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
    g->whiteTex = g->device.CreateTexture(&td);
    g->whiteView = g->whiteTex.CreateView();
    uint32_t white = 0xffffffffu;
    wgpu::TexelCopyTextureInfo dst{};
    dst.texture = g->whiteTex;
    wgpu::TexelCopyBufferLayout lay{};
    lay.bytesPerRow = 4;
    wgpu::Extent3D ext{1, 1, 1};
    g->queue.WriteTexture(&dst, &white, 4, &lay, &ext);
}

/* Read a region of the colour target, GL order (bottom row first). */
void readPixels(GLint x, GLint y, GLsizei w, GLsizei h, uint8_t* out) {
    if (w <= 0 || h <= 0) return;
    submit();
    uint32_t top = static_cast<uint32_t>(g->fbH - y - h);
    uint32_t bpr = (static_cast<uint32_t>(w) * 4 + 255) & ~255u;
    wgpu::BufferDescriptor bd{};
    bd.size = static_cast<uint64_t>(bpr) * h;
    bd.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    wgpu::Buffer buf = g->device.CreateBuffer(&bd);
    wgpu::CommandEncoder enc = g->device.CreateCommandEncoder();
    wgpu::TexelCopyTextureInfo src{};
    src.texture = g->color;
    src.origin = {static_cast<uint32_t>(x), top, 0};
    wgpu::TexelCopyBufferInfo dst{};
    dst.buffer = buf;
    dst.layout.bytesPerRow = bpr;
    dst.layout.rowsPerImage = h;
    wgpu::Extent3D ext{static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
    enc.CopyTextureToBuffer(&src, &dst, &ext);
    wgpu::CommandBuffer cb = enc.Finish();
    g->queue.Submit(1, &cb);
    bool ok = false;
    wgpu::Future f = buf.MapAsync(wgpu::MapMode::Read, 0, bd.size, wgpu::CallbackMode::WaitAnyOnly,
                                  [&](wgpu::MapAsyncStatus s, wgpu::StringView) { ok = s == wgpu::MapAsyncStatus::Success; });
    g->instance.WaitAny(f, UINT64_MAX);
    if (!ok) return;
    const uint8_t* p = static_cast<const uint8_t*>(buf.GetConstMappedRange(0, bd.size));
    for (int r = 0; r < h; r++)
        memcpy(out + static_cast<size_t>(r) * w * 4, p + static_cast<size_t>(h - 1 - r) * bpr, static_cast<size_t>(w) * 4);
    buf.Unmap();
}

void dumpFrame() {
    std::vector<uint8_t> px(static_cast<size_t>(g->fbW) * g->fbH * 4);
    readPixels(0, 0, g->fbW, g->fbH, px.data());
    char path[512];
    snprintf(path, sizeof path, "%s/frame%06llu.ppm", g->dumpDir.c_str(), static_cast<unsigned long long>(g->frame));
    FILE* f = fopen(path, "wb");
    if (!f) return;
    fprintf(f, "P6\n%d %d\n255\n", g->fbW, g->fbH);
    for (int y = g->fbH - 1; y >= 0; y--)
        for (int x = 0; x < g->fbW; x++) fwrite(&px[(static_cast<size_t>(y) * g->fbW + x) * 4], 1, 3, f);
    fclose(f);
}

/* ------------------------------------------------------------------ */
/* GL entry points                                                     */
/* ------------------------------------------------------------------ */

Program* currentProgram() {
    auto it = g->programs.find(g->currentProgram);
    return it == g->programs.end() ? nullptr : it->second.get();
}

void setUniformRaw(Program& p, int index, int element, const void* data, size_t bytes) {
    Uniform& u = p.uniforms[index];
    for (int stage = 0; stage < 2; stage++) {
        if (u.offset[stage] < 0) continue;
        size_t at = static_cast<size_t>(u.offset[stage]) + static_cast<size_t>(element) * u.arrayStride;
        if (at + bytes <= p.block[stage].size()) memcpy(&p.block[stage][at], data, bytes);
    }
}

/* Write `count` array elements of `comps` 4-byte components each. */
void setUniform(GLint location, GLsizei count, int comps, const void* data) {
    Program* p = currentProgram();
    if (!p || location < 0) return;
    int index = location >> 8, element = location & 0xff;
    if (index >= static_cast<int>(p->uniforms.size())) return;
    Uniform& u = p->uniforms[index];
    if (u.sampler >= 0) {
        p->samplerUnit[u.sampler] = *static_cast<const GLint*>(data);
        return;
    }
    const uint8_t* src = static_cast<const uint8_t*>(data);
    for (int i = 0; i < count && element + i < u.arraySize; i++)
        setUniformRaw(*p, index, element + i, src + static_cast<size_t>(i) * comps * 4, static_cast<size_t>(comps) * 4);
}

void setMatrix(GLint location, GLsizei count, int dim, GLboolean transpose, const GLfloat* v) {
    Program* p = currentProgram();
    if (!p || location < 0) return;
    int index = location >> 8;
    if (index >= static_cast<int>(p->uniforms.size())) return;
    for (int i = 0; i < count; i++) {
        float cols[16] = {};                      /* std140: each column a vec4 */
        for (int c = 0; c < dim; c++)
            for (int r = 0; r < dim; r++)
                cols[c * 4 + r] = transpose ? v[i * dim * dim + r * dim + c] : v[i * dim * dim + c * dim + r];
        setUniformRaw(*p, index, (location & 0xff) + i, cols, static_cast<size_t>(dim) * 16);
    }
}

#define GLAPI extern "C"

GLAPI void glwgpu_glGenTextures(GLsizei n, GLuint* ids) { for (int i = 0; i < n; i++) { ids[i] = g->nextName++; g->textures[ids[i]]; } }
GLAPI void glwgpu_glDeleteTextures(GLsizei n, const GLuint* ids) {
    for (int i = 0; i < n; i++) {
        g->textures.erase(ids[i]);
        for (auto& t : g->unitTexture) if (t == ids[i]) t = 0;
    }
}
GLAPI void glwgpu_glActiveTexture(GLenum unit) { g->activeUnit = static_cast<int>(unit - GL_TEXTURE0) & 15; }
GLAPI void glwgpu_glBindTexture(GLenum, GLuint id) {
    g->unitTexture[g->activeUnit] = id;
    if (id) g->textures[id];
}
GLAPI void glwgpu_glTexParameteri(GLenum, GLenum pname, GLint v) {
    Texture* t = textureForUnit(g->activeUnit);
    if (!t) {
        auto it = g->textures.find(g->unitTexture[g->activeUnit]);
        if (it == g->textures.end()) return;
        t = &it->second;
    }
    if (pname == GL_TEXTURE_MIN_FILTER) t->minFilter = v;
    else if (pname == GL_TEXTURE_MAG_FILTER) t->magFilter = v;
    else if (pname == GL_TEXTURE_WRAP_S) t->wrapS = v;
    else if (pname == GL_TEXTURE_WRAP_T) t->wrapT = v;
}
GLAPI void glwgpu_glPixelStorei(GLenum, GLint) {}
GLAPI void glwgpu_glTexImage2D(GLenum, GLint level, GLint, GLsizei w, GLsizei h, GLint, GLenum, GLenum,
                               const void* pixels) {
    if (level != 0 || w <= 0 || h <= 0) return;
    GLuint id = g->unitTexture[g->activeUnit];
    if (!id) return;
    Texture& t = g->textures[id];
    if (!t.tex || t.w != w || t.h != h || t.usedSerial == g->batch) {
        wgpu::TextureDescriptor d{};
        d.size = {static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
        d.format = wgpu::TextureFormat::RGBA8Unorm;
        d.usage = wgpu::TextureUsage::TextureBinding | wgpu::TextureUsage::CopyDst;
        t.tex = g->device.CreateTexture(&d);
        t.view = t.tex.CreateView();
        t.w = w;
        t.h = h;
        t.usedSerial = 0;
    }
    if (!pixels) return;
    wgpu::TexelCopyTextureInfo dst{};
    dst.texture = t.tex;
    wgpu::TexelCopyBufferLayout lay{};
    lay.bytesPerRow = static_cast<uint32_t>(w) * 4;
    lay.rowsPerImage = h;
    wgpu::Extent3D ext{static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
    g->queue.WriteTexture(&dst, pixels, static_cast<size_t>(w) * h * 4, &lay, &ext);
}

GLAPI void glwgpu_glGenBuffers(GLsizei n, GLuint* ids) { for (int i = 0; i < n; i++) { ids[i] = g->nextName++; g->buffers[ids[i]]; } }
GLAPI void glwgpu_glDeleteBuffers(GLsizei n, const GLuint* ids) { for (int i = 0; i < n; i++) g->buffers.erase(ids[i]); }
GLAPI void glwgpu_glBindBuffer(GLenum target, GLuint id) {
    if (target == GL_ARRAY_BUFFER) g->boundArray = id;
    else if (target == GL_ELEMENT_ARRAY_BUFFER) g->vaos[g->boundVao].elementBuffer = id;
}
GLAPI void glwgpu_glBufferData(GLenum target, GLsizeiptr size, const void* data, GLenum) {
    GLuint id = target == GL_ARRAY_BUFFER ? g->boundArray : g->vaos[g->boundVao].elementBuffer;
    Buffer& b = g->buffers[id];
    if (data) b.data.assign(static_cast<const uint8_t*>(data), static_cast<const uint8_t*>(data) + size);
    else b.data.assign(static_cast<size_t>(size), 0);
    b.gpuDirty = true;
}
GLAPI void glwgpu_glGenVertexArrays(GLsizei n, GLuint* ids) { for (int i = 0; i < n; i++) { ids[i] = g->nextName++; g->vaos[ids[i]]; } }
GLAPI void glwgpu_glDeleteVertexArrays(GLsizei n, const GLuint* ids) { for (int i = 0; i < n; i++) g->vaos.erase(ids[i]); }
GLAPI void glwgpu_glBindVertexArray(GLuint id) { g->boundVao = id; }
GLAPI void glwgpu_glEnableVertexAttribArray(GLuint i) { if (i < 8) g->vaos[g->boundVao].attr[i].enabled = true; }
GLAPI void glwgpu_glVertexAttribPointer(GLuint i, GLint size, GLenum type, GLboolean norm, GLsizei stride, const void* ptr) {
    if (i >= 8) return;
    Attrib& a = g->vaos[g->boundVao].attr[i];
    a.size = size;
    a.type = type;
    a.normalized = norm;
    a.stride = stride;
    a.offset = reinterpret_cast<uintptr_t>(ptr);
    a.buffer = g->boundArray;
}

GLAPI GLuint glwgpu_glCreateShader(GLenum type) { GLuint id = g->nextName++; g->shaders[id] = {type, {}}; return id; }
GLAPI void glwgpu_glDeleteShader(GLuint id) { (void)id; /* programs may still link against it */ }
GLAPI void glwgpu_glShaderSource(GLuint id, GLsizei n, const GLchar* const* strs, const GLint* lens) {
    std::string s;
    for (int i = 0; i < n; i++) s += lens && lens[i] >= 0 ? std::string(strs[i], lens[i]) : std::string(strs[i]);
    g->shaders[id].source = s;
}
GLAPI void glwgpu_glCompileShader(GLuint) {}
GLAPI void glwgpu_glGetShaderiv(GLuint, GLenum pname, GLint* v) { *v = pname == GL_COMPILE_STATUS ? 1 : 0; }
GLAPI void glwgpu_glGetShaderInfoLog(GLuint, GLsizei max, GLsizei* len, GLchar* log) { if (len) *len = 0; if (max) log[0] = 0; }
GLAPI GLuint glwgpu_glCreateProgram(void) { GLuint id = g->nextName++; g->programs[id] = std::make_unique<Program>(); return id; }
GLAPI void glwgpu_glDeleteProgram(GLuint id) { g->programs.erase(id); }
GLAPI void glwgpu_glAttachShader(GLuint prog, GLuint sh) {
    auto& p = *g->programs[prog];
    (g->shaders[sh].type == GL_VERTEX_SHADER ? p.vs : p.fs) = sh;
}
GLAPI void glwgpu_glBindAttribLocation(GLuint, GLuint, const GLchar*) {}
GLAPI void glwgpu_glLinkProgram(GLuint id) { linkProgram(*g->programs[id]); }
GLAPI void glwgpu_glGetProgramiv(GLuint id, GLenum pname, GLint* v) {
    *v = pname == GL_LINK_STATUS ? (g->programs[id]->linked ? 1 : 0) : 0;
}
GLAPI void glwgpu_glGetProgramInfoLog(GLuint, GLsizei max, GLsizei* len, GLchar* log) { if (len) *len = 0; if (max) log[0] = 0; }
GLAPI void glwgpu_glUseProgram(GLuint id) { g->currentProgram = id; }
GLAPI GLint glwgpu_glGetUniformLocation(GLuint id, const GLchar* name) {
    auto it = g->programs.find(id);
    if (it == g->programs.end()) return -1;
    Program& p = *it->second;
    std::string n(name);
    int element = 0;
    size_t br = n.find('[');
    if (br != std::string::npos) {
        element = atoi(n.c_str() + br + 1);
        n = n.substr(0, br);
    }
    auto u = p.byName.find(n);
    if (u == p.byName.end() || element > 255) return -1;
    return (u->second << 8) | element;
}
GLAPI void glwgpu_glUniform1i(GLint l, GLint a) { setUniform(l, 1, 1, &a); }
GLAPI void glwgpu_glUniform2i(GLint l, GLint a, GLint b) { GLint v[2] = {a, b}; setUniform(l, 1, 2, v); }
GLAPI void glwgpu_glUniform3i(GLint l, GLint a, GLint b, GLint c) { GLint v[3] = {a, b, c}; setUniform(l, 1, 3, v); }
GLAPI void glwgpu_glUniform4i(GLint l, GLint a, GLint b, GLint c, GLint d) { GLint v[4] = {a, b, c, d}; setUniform(l, 1, 4, v); }
GLAPI void glwgpu_glUniform1f(GLint l, GLfloat a) { setUniform(l, 1, 1, &a); }
GLAPI void glwgpu_glUniform2f(GLint l, GLfloat a, GLfloat b) { GLfloat v[2] = {a, b}; setUniform(l, 1, 2, v); }
GLAPI void glwgpu_glUniform3f(GLint l, GLfloat a, GLfloat b, GLfloat c) { GLfloat v[3] = {a, b, c}; setUniform(l, 1, 3, v); }
GLAPI void glwgpu_glUniform4f(GLint l, GLfloat a, GLfloat b, GLfloat c, GLfloat d) { GLfloat v[4] = {a, b, c, d}; setUniform(l, 1, 4, v); }
GLAPI void glwgpu_glUniform1iv(GLint l, GLsizei n, const GLint* v) {
    Program* p = currentProgram();
    if (p && l >= 0 && (l >> 8) < static_cast<int>(p->uniforms.size()) && p->uniforms[l >> 8].sampler >= 0) {
        p->samplerUnit[p->uniforms[l >> 8].sampler] = v[0];
        return;
    }
    setUniform(l, n, 1, v);
}
GLAPI void glwgpu_glUniform2iv(GLint l, GLsizei n, const GLint* v) { setUniform(l, n, 2, v); }
GLAPI void glwgpu_glUniform3iv(GLint l, GLsizei n, const GLint* v) { setUniform(l, n, 3, v); }
GLAPI void glwgpu_glUniform4iv(GLint l, GLsizei n, const GLint* v) { setUniform(l, n, 4, v); }
GLAPI void glwgpu_glUniform1fv(GLint l, GLsizei n, const GLfloat* v) { setUniform(l, n, 1, v); }
GLAPI void glwgpu_glUniform2fv(GLint l, GLsizei n, const GLfloat* v) { setUniform(l, n, 2, v); }
GLAPI void glwgpu_glUniform3fv(GLint l, GLsizei n, const GLfloat* v) { setUniform(l, n, 3, v); }
GLAPI void glwgpu_glUniform4fv(GLint l, GLsizei n, const GLfloat* v) { setUniform(l, n, 4, v); }
GLAPI void glwgpu_glUniformMatrix3fv(GLint l, GLsizei n, GLboolean t, const GLfloat* v) { setMatrix(l, n, 3, t, v); }
GLAPI void glwgpu_glUniformMatrix4fv(GLint l, GLsizei n, GLboolean t, const GLfloat* v) { setMatrix(l, n, 4, t, v); }

GLAPI void glwgpu_glEnable(GLenum cap) {
    if (cap == GL_BLEND) g->blend = true;
    else if (cap == GL_CULL_FACE) g->cull = true;
    else if (cap == GL_DEPTH_TEST) g->depthTest = true;
    else if (cap == GL_SCISSOR_TEST) g->scissorTest = true;
}
GLAPI void glwgpu_glDisable(GLenum cap) {
    if (cap == GL_BLEND) g->blend = false;
    else if (cap == GL_CULL_FACE) g->cull = false;
    else if (cap == GL_DEPTH_TEST) g->depthTest = false;
    else if (cap == GL_SCISSOR_TEST) g->scissorTest = false;
}
GLAPI void glwgpu_glBlendFunc(GLenum s, GLenum d) { g->blendSrc = s; g->blendDst = d; }
GLAPI void glwgpu_glBlendEquation(GLenum e) { g->blendEq = e; }
GLAPI void glwgpu_glCullFace(GLenum f) { g->cullFace = f; }
GLAPI void glwgpu_glDepthFunc(GLenum f) { g->depthFunc = f; }
GLAPI void glwgpu_glDepthMask(GLboolean m) { g->depthMask = m != 0; }
GLAPI void glwgpu_glColorMask(GLboolean r, GLboolean gg, GLboolean b, GLboolean a) {
    g->colorMask = (r ? 1 : 0) | (gg ? 2 : 0) | (b ? 4 : 0) | (a ? 8 : 0);
}
GLAPI void glwgpu_glDepthRange(GLdouble n, GLdouble f) { g->depthNear = static_cast<float>(n); g->depthFar = static_cast<float>(f); }
GLAPI void glwgpu_glViewport(GLint x, GLint y, GLsizei w, GLsizei h) { g->vp[0] = x; g->vp[1] = y; g->vp[2] = w; g->vp[3] = h; }
GLAPI void glwgpu_glScissor(GLint x, GLint y, GLsizei w, GLsizei h) { g->sc[0] = x; g->sc[1] = y; g->sc[2] = w; g->sc[3] = h; }
GLAPI void glwgpu_glClearColor(GLfloat r, GLfloat gg, GLfloat b, GLfloat a) {
    g->clearColor[0] = r; g->clearColor[1] = gg; g->clearColor[2] = b; g->clearColor[3] = a;
}
GLAPI void glwgpu_glClearDepth(GLdouble d) { g->clearDepth = static_cast<float>(d); }
GLAPI void glwgpu_glClear(GLbitfield mask) { clear(mask); }
GLAPI void glwgpu_glPointSize(GLfloat) {}
GLAPI void glwgpu_glLineWidth(GLfloat) {}
GLAPI void glwgpu_glFlush(void) {}
GLAPI void glwgpu_glFinish(void) { submit(); waitIdle(); }
GLAPI GLenum glwgpu_glGetError(void) { return GL_NO_ERROR; }
GLAPI void glwgpu_glDrawArrays(GLenum mode, GLint first, GLsizei count) { draw(mode, first, count, 0, 0, false); }
GLAPI void glwgpu_glDrawElements(GLenum mode, GLsizei count, GLenum type, const void* offset) {
    draw(mode, 0, count, type, reinterpret_cast<uintptr_t>(offset), true);
}
GLAPI void glwgpu_glReadPixels(GLint x, GLint y, GLsizei w, GLsizei h, GLenum, GLenum, void* out) {
    readPixels(x, y, w, h, static_cast<uint8_t*>(out));
}
GLAPI const GLubyte* glwgpu_glGetString(GLenum name) {
    switch (name) {
    case GL_VENDOR: return reinterpret_cast<const GLubyte*>("glwgpu");
    case GL_RENDERER: return reinterpret_cast<const GLubyte*>("glwgpu (Dawn/WebGPU)");
    case GL_VERSION: return reinterpret_cast<const GLubyte*>("3.3.0 glwgpu");
    case GL_SHADING_LANGUAGE_VERSION: return reinterpret_cast<const GLubyte*>("3.30");
    default: return reinterpret_cast<const GLubyte*>("");
    }
}
GLAPI const GLubyte* glwgpu_glGetStringi(GLenum, GLuint) { return nullptr; }
GLAPI void glwgpu_glGetIntegerv(GLenum pname, GLint* v) {
    switch (pname) {
    case GL_NUM_EXTENSIONS: *v = 0; break;
    case GL_MAX_TEXTURE_SIZE: *v = 8192; break;
    case GL_MAJOR_VERSION: *v = 3; break;
    case GL_MINOR_VERSION: *v = 3; break;
    default: *v = 0; break;
    }
}

struct ProcEntry { const char* name; void* fn; };
#define P(n) {#n, reinterpret_cast<void*>(&glwgpu_##n)}
const ProcEntry kProcs[] = {
    P(glGenTextures), P(glDeleteTextures), P(glActiveTexture), P(glBindTexture), P(glTexParameteri),
    P(glPixelStorei), P(glTexImage2D), P(glGenBuffers), P(glDeleteBuffers), P(glBindBuffer), P(glBufferData),
    P(glGenVertexArrays), P(glDeleteVertexArrays), P(glBindVertexArray), P(glEnableVertexAttribArray),
    P(glVertexAttribPointer), P(glCreateShader), P(glDeleteShader), P(glShaderSource), P(glCompileShader),
    P(glGetShaderiv), P(glGetShaderInfoLog), P(glCreateProgram), P(glDeleteProgram), P(glAttachShader),
    P(glBindAttribLocation), P(glLinkProgram), P(glGetProgramiv), P(glGetProgramInfoLog), P(glUseProgram),
    P(glGetUniformLocation), P(glUniform1i), P(glUniform2i), P(glUniform3i), P(glUniform4i), P(glUniform1f),
    P(glUniform2f), P(glUniform3f), P(glUniform4f), P(glUniform1iv), P(glUniform2iv), P(glUniform3iv),
    P(glUniform4iv), P(glUniform1fv), P(glUniform2fv), P(glUniform3fv), P(glUniform4fv),
    P(glUniformMatrix3fv), P(glUniformMatrix4fv), P(glEnable), P(glDisable), P(glBlendFunc),
    P(glBlendEquation), P(glCullFace), P(glDepthFunc), P(glDepthMask), P(glColorMask), P(glDepthRange),
    P(glViewport), P(glScissor), P(glClearColor), P(glClearDepth), P(glClear), P(glPointSize), P(glLineWidth),
    P(glFlush), P(glFinish), P(glGetError), P(glDrawArrays), P(glDrawElements), P(glReadPixels),
    P(glGetString), P(glGetStringi), P(glGetIntegerv),
};
#undef P

void present() {
    submit();
    g->frame++;
    if (!g->dumpDir.empty() && g->frame % g->dumpEvery == 0) dumpFrame();
    if (!g->surface) return;
    wgpu::SurfaceTexture st;
    g->surface.GetCurrentTexture(&st);
    if (st.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessOptimal &&
        st.status != wgpu::SurfaceGetCurrentTextureStatus::SuccessSuboptimal) {
        static int reported;
        if (reported++ < 8) logf("frame %llu: GetCurrentTexture status %d", static_cast<unsigned long long>(g->frame),
                                 static_cast<int>(st.status));
        return;
    }
    if (g->frame == 1 || g->frame % 600 == 0)
        logf("presented frame %llu (%ux%u surface)", static_cast<unsigned long long>(g->frame),
             st.texture.GetWidth(), st.texture.GetHeight());
    /* Fit the EFB into the surface keeping its aspect (the port renders
     * widescreen itself when enabled, so the EFB size carries the aspect). */
    wgpu::TextureView target = st.texture.CreateView();
    float sw = static_cast<float>(g->surfaceW), sh = static_cast<float>(g->surfaceH);
    float scale = std::min(sw / g->fbW, sh / g->fbH);
    float w = g->fbW * scale, h = g->fbH * scale;
    wgpu::CommandEncoder enc = g->device.CreateCommandEncoder();
    wgpu::RenderPassColorAttachment ca{};
    ca.view = target;
    ca.loadOp = wgpu::LoadOp::Clear;
    ca.storeOp = wgpu::StoreOp::Store;
    ca.clearValue = {0, 0, 0, 1};
    wgpu::RenderPassDescriptor rp{};
    rp.colorAttachmentCount = 1;
    rp.colorAttachments = &ca;
    wgpu::RenderPassEncoder pass = enc.BeginRenderPass(&rp);
    wgpu::BindGroupEntry be[2]{};
    be[0].binding = 0;
    be[0].textureView = g->colorView;
    be[1].binding = 1;
    be[1].sampler = g->blitSampler;
    wgpu::BindGroupDescriptor bd{};
    bd.layout = g->blitBgl;
    bd.entryCount = 2;
    bd.entries = be;
    wgpu::BindGroup bg = g->device.CreateBindGroup(&bd);
    pass.SetPipeline(g->blitPipeline);
    pass.SetBindGroup(0, bg);
    pass.SetViewport((sw - w) * 0.5f, (sh - h) * 0.5f, w, h, 0, 1);
    pass.Draw(3);
    pass.End();
    wgpu::CommandBuffer cb = enc.Finish();
    g->queue.Submit(1, &cb);
    g->surface.Present();
}

void createBlit() {
    const char* wgsl = R"(
@group(0) @binding(0) var t : texture_2d<f32>;
@group(0) @binding(1) var s : sampler;
struct V { @builtin(position) p : vec4f, @location(0) uv : vec2f };
@vertex fn vs(@builtin(vertex_index) i : u32) -> V {
  let p = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o : V;
  o.p = vec4f(p[i], 0.0, 1.0);
  o.uv = vec2f(p[i].x * 0.5 + 0.5, 0.5 - p[i].y * 0.5);
  return o;
}
@fragment fn fs(v : V) -> @location(0) vec4f { return vec4f(textureSample(t, s, v.uv).rgb, 1.0); }
)";
    wgpu::ShaderSourceWGSL w{};
    w.code = wgsl;
    wgpu::ShaderModuleDescriptor md{};
    md.nextInChain = &w;
    wgpu::ShaderModule m = g->device.CreateShaderModule(&md);
    wgpu::ColorTargetState ct{};
    ct.format = g->surfaceFormat;
    wgpu::FragmentState fs{};
    fs.module = m;
    fs.entryPoint = "fs";
    fs.targetCount = 1;
    fs.targets = &ct;
    wgpu::RenderPipelineDescriptor d{};
    d.vertex.module = m;
    d.vertex.entryPoint = "vs";
    d.fragment = &fs;
    g->blitPipeline = g->device.CreateRenderPipeline(&d);
    g->blitBgl = g->blitPipeline.GetBindGroupLayout(0);
    wgpu::SamplerDescriptor sd{};
    sd.magFilter = wgpu::FilterMode::Linear;
    sd.minFilter = wgpu::FilterMode::Linear;
    g->blitSampler = g->device.CreateSampler(&sd);
}

}  // namespace

extern "C" int glwgpu_init(const glwgpu_config* config) {
    g = new Context();
    g->fbW = config->width > 0 ? config->width : 640;
    g->fbH = config->height > 0 ? config->height : 480;
    g->vp[2] = g->sc[2] = g->fbW;
    g->vp[3] = g->sc[3] = g->fbH;
    if (const char* d = getenv("GLWGPU_DUMP_DIR")) g->dumpDir = d;
    if (const char* e = getenv("GLWGPU_DUMP_EVERY")) g->dumpEvery = std::max(1, atoi(e));
    glslang::InitializeProcess();

    std::array<wgpu::InstanceFeatureName, 2> features{wgpu::InstanceFeatureName::TimedWaitAny,
                                                      wgpu::InstanceFeatureName::ShaderSourceSPIRV};
    wgpu::InstanceDescriptor id{};
    id.requiredFeatureCount = features.size();
    id.requiredFeatures = features.data();
    g->instance = wgpu::CreateInstance(&id);
    if (!g->instance) {
        logf("no WebGPU instance");
        return 0;
    }
#ifdef HBX_SWITCH2
    if (config->switch_vi_window) {
        wgpu::SurfaceSourceSwitchNWindow src{};
        src.window = config->switch_vi_window;
        wgpu::SurfaceDescriptor sd{};
        sd.nextInChain = &src;
        g->surface = g->instance.CreateSurface(&sd);
    }
#endif
    wgpu::RequestAdapterOptions ao{};
    ao.powerPreference = wgpu::PowerPreference::HighPerformance;
    ao.compatibleSurface = g->surface;
    auto fa = g->instance.RequestAdapter(&ao, wgpu::CallbackMode::WaitAnyOnly,
                                         [](wgpu::RequestAdapterStatus s, wgpu::Adapter a, wgpu::StringView m) {
                                             if (s == wgpu::RequestAdapterStatus::Success) g->adapter = std::move(a);
                                             else logf("adapter: %.*s", static_cast<int>(m.length), m.data);
                                         });
    g->instance.WaitAny(fa, UINT64_MAX);
    if (!g->adapter) return 0;
    wgpu::DeviceDescriptor dd{};
    dd.SetUncapturedErrorCallback([](const wgpu::Device&, wgpu::ErrorType type, wgpu::StringView msg) {
        logf("WebGPU error %d: %.*s", static_cast<int>(type), static_cast<int>(msg.length), msg.data);
    });
    auto fd = g->adapter.RequestDevice(&dd, wgpu::CallbackMode::WaitAnyOnly,
                                       [](wgpu::RequestDeviceStatus s, wgpu::Device d, wgpu::StringView m) {
                                           if (s == wgpu::RequestDeviceStatus::Success) g->device = std::move(d);
                                           else logf("device: %.*s", static_cast<int>(m.length), m.data);
                                       });
    g->instance.WaitAny(fd, UINT64_MAX);
    if (!g->device) return 0;
    g->queue = g->device.GetQueue();
    wgpu::Limits limits{};
    g->device.GetLimits(&limits);
    g->uniformAlign = limits.minUniformBufferOffsetAlignment;

    if (g->surface) {
        wgpu::SurfaceCapabilities caps{};
        g->surface.GetCapabilities(g->adapter, &caps);
        g->surfaceFormat = caps.formatCount ? caps.formats[0] : wgpu::TextureFormat::RGBA8Unorm;
        for (size_t i = 0; i < caps.formatCount; i++)
            if (caps.formats[i] == wgpu::TextureFormat::RGBA8Unorm) g->surfaceFormat = caps.formats[i];
        g->surfaceW = 1920;
        g->surfaceH = 1080;
        wgpu::SurfaceConfiguration sc{};
        sc.device = g->device;
        sc.format = g->surfaceFormat;
        sc.width = g->surfaceW;
        sc.height = g->surfaceH;
        sc.presentMode = wgpu::PresentMode::Fifo;
        g->surface.Configure(&sc);
        logf("surface: format %d, %zu formats, %zu present modes", static_cast<int>(g->surfaceFormat),
             caps.formatCount, caps.presentModeCount);
        createBlit();
    }

    createFramebuffer();
    makeRing(g->vertexRing, 32u << 20, wgpu::BufferUsage::Vertex);
    makeRing(g->uniformRing, 16u << 20, wgpu::BufferUsage::Uniform);
    makeRing(g->indexRing, 4u << 20, wgpu::BufferUsage::Index);
    g->vaos[0];
    g->buffers[0];
    createHelpers();
    logf("ready: %dx%d framebuffer, %s", g->fbW, g->fbH, g->surface ? "surface" : "headless");
    return 1;
}

extern "C" void* glwgpu_get_proc(const char* name) {
    for (const auto& e : kProcs)
        if (strcmp(e.name, name) == 0) return e.fn;
    return nullptr;
}

extern "C" void glwgpu_present(void) { present(); }

extern "C" void glwgpu_resize(int width, int height) {
    if (width <= 0 || height <= 0 || (width == g->fbW && height == g->fbH)) return;
    submit();
    g->fbW = width;
    g->fbH = height;
    createFramebuffer();
}

extern "C" void glwgpu_shutdown(void) {
    if (!g) return;
    submit();
    glslang::FinalizeProcess();
    /* The instance is deliberately kept: destroying it crashes on the
     * Switch (HBLauncher task #13). */
}
