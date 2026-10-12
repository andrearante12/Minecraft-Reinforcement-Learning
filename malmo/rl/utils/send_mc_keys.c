/* Send F5 (third person) and/or F1 (hide GUI) to a window titled Minecraft. */
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>

/* libXtst is present; headers may not be. */
extern int XTestFakeKeyEvent(Display *, unsigned int, int, unsigned long);

static int name_has_minecraft(Display *dpy, Window w) {
    char *name = NULL;
    if (XFetchName(dpy, w, &name) && name) {
        int hit = strstr(name, "Minecraft") != NULL;
        XFree(name);
        return hit;
    }
    return 0;
}

static Window find_minecraft(Display *dpy, Window w) {
    Window root, parent, *kids = NULL;
    unsigned n = 0;
    if (!XQueryTree(dpy, w, &root, &parent, &kids, &n))
        return 0;
    Window found = 0;
    for (unsigned i = 0; i < n && !found; i++)
        found = find_minecraft(dpy, kids[i]);
    if (kids)
        XFree(kids);
    if (found)
        return found;
    return name_has_minecraft(dpy, w) ? w : 0;
}

static void tap(Display *dpy, KeySym sym) {
    KeyCode code = XKeysymToKeycode(dpy, sym);
    if (!code)
        return;
    XTestFakeKeyEvent(dpy, code, True, CurrentTime);
    XFlush(dpy);
    usleep(40000);
    XTestFakeKeyEvent(dpy, code, False, CurrentTime);
    XFlush(dpy);
    usleep(80000);
}

int main(int argc, char **argv) {
    int do_f5 = 0, do_f1 = 0;
    int i;
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "F5") == 0) do_f5 = 1;
        if (strcmp(argv[i], "F1") == 0) do_f1 = 1;
    }
    if (!do_f5 && !do_f1) {
        do_f5 = do_f1 = 1;
    }

    Display *dpy = XOpenDisplay(NULL);
    if (!dpy) {
        fprintf(stderr, "send_mc_keys: no display\n");
        return 1;
    }
    Window found = find_minecraft(dpy, DefaultRootWindow(dpy));
    if (!found) {
        fprintf(stderr, "send_mc_keys: no Minecraft window\n");
        XCloseDisplay(dpy);
        return 2;
    }
    XMapRaised(dpy, found);
    XSetInputFocus(dpy, found, RevertToParent, CurrentTime);
    XFlush(dpy);
    usleep(200000);
    if (do_f5) {
        tap(dpy, XK_F5);
        fprintf(stderr, "send_mc_keys: F5 -> 0x%lx\n", (unsigned long)found);
    }
    if (do_f1) {
        tap(dpy, XK_F1);
        fprintf(stderr, "send_mc_keys: F1 -> 0x%lx\n", (unsigned long)found);
    }
    XCloseDisplay(dpy);
    return 0;
}
