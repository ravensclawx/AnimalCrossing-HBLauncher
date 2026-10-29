/*
 * glwgpu - the OpenGL 3.3 subset the Animal Crossing PC port's renderer uses,
 * implemented on WebGPU (Dawn).  The port's GX layer (pc_gx*.c) runs
 * unchanged: gladLoadGL() is pointed at glwgpu_get_proc instead of a real GL
 * context.  Shaders are GLSL 330, compiled with glslang (relaxed Vulkan
 * rules) to SPIR-V for Dawn.
 */
#ifndef GLWGPU_H
#define GLWGPU_H

#ifdef __cplusplus
extern "C" {
#endif

/* Window/surface the frames go to.  NULL native window = headless: frames
 * stay offscreen (GLWGPU_DUMP_DIR=<dir> dumps every GLWGPU_DUMP_EVERY'th
 * frame as PPM). */
typedef struct glwgpu_config {
    int width, height;              /* default framebuffer (the GX EFB) */
    void* switch_vi_window;         /* HBLauncher Switch 2: nn::vi window */
    void* (*get_vk_proc)(void* instance, const char* name); /* unused on PC */
} glwgpu_config;

int glwgpu_init(const glwgpu_config* config);
void* glwgpu_get_proc(const char* name);
void glwgpu_present(void);
void glwgpu_resize(int width, int height);
void glwgpu_shutdown(void);

#ifdef __cplusplus
}
#endif
#endif
