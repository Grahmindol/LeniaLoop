//#include <GL/glut.h>
#include <GL/freeglut.h>
#include <stdio.h>

#include "headers/life.h"

int frameCount = 0;
double lastTime = 0.0;
int offsetX = 0, offsetY = 0;
long frameCountsincestart = 0;
double med_fps = 60;
double zoom = 1024.0/GRID_SIZE;  // Global zoom factor

// OpenGL initialization for 2D rendering
void initOpenGL(void)
{
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    // Set up an orthographic projection with the grid centered
    gluOrtho2D(-GRID_SIZE / 2, GRID_SIZE / 2, -GRID_SIZE / 2, GRID_SIZE / 2);
}

// Display the current FPS in the window title.
void displayFPS(void)
{
    frameCount++;
     
    double currentTime = glutGet(GLUT_ELAPSED_TIME) / 1000.0; // seconds
    double deltaTime = currentTime - lastTime;

    if (deltaTime >= 1.0) {
        char title[50];
        double fps = frameCount / deltaTime;
        med_fps = ((med_fps*frameCountsincestart +  fps)/ (frameCountsincestart + 1.0));
        frameCountsincestart++;
        snprintf(title, sizeof(title), "OpenGL FPS: %.0f / %.2f", fps ,med_fps);
        glutSetWindowTitle(title);
        frameCount = 0; 
        lastTime = currentTime;
    }
}

// Render the current frame using the current pixel buffer.
void display(void)
{
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Apply zoom for glDrawPixels
    glPixelZoom(zoom, zoom);
    
    
    glRasterPos2i(-GRID_SIZE / 2, -GRID_SIZE / 2);
    int currentFrame = frameCount & 1;
    glDrawPixels(GRID_SIZE, GRID_SIZE, GL_RGB, GL_FLOAT, pixels); // does not work when raster pos is negtive
    glutSwapBuffers();
}

// The main loop function that updates the state, FPS, and triggers a redraw.
void looply(int value)
{
    life_update_frame(value);
    displayFPS();
    #ifdef DISPLAY
    glutPostRedisplay();
    #endif
    glutTimerFunc(0, looply, (value+1)%BUFFER_NUMBER);
}

// Handle mouse wheel for zooming.
// Note: Many GLUT implementations treat mouse wheel events as buttons 3 (wheel up) and 4 (wheel down).
void mouseWheel(int button, int state, int x, int y)
{
    frameCount ++;
    life_update_frame(frameCount % BUFFER_NUMBER);
    glutPostRedisplay();
}

void cleanup(){
    destroy_convolve_buffer();
    destroy_life();
}

int main(int argc, char **argv)
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(1024, 1024);
    glutCreateWindow("OpenGL PNG Loader");

    initOpenGL();
    
    init_life();  // Load the image into the grid

    // Register GLUT callbacks
    glutDisplayFunc(display);

    #ifndef STEP_BY_STEP_MOD
    glutTimerFunc(0, looply, 0);
    #else
    life_update_frame(0);
    glutMouseFunc(mouseWheel);
    #endif
    // Register cleanup function to run on exit
    atexit(cleanup);

    glutMainLoop();

    
    return 0;
}
