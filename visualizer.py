#!/usr/bin/env python3

import numpy as np

from PIL import Image, ImageTk
import tkinter as tk
import aux_info
import torch
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import utils

def calc_length(x):
    return np.sqrt(x.dot(x))


class BRDFviewer:
    def __init__(self, model, use_pathguiding):



        self.model = model
        self.patch_size = self.model.resolution
        self.old_state = None
        self.data = None
        self.data_prob = None
        self.use_pathguiding = use_pathguiding







    def eval(self, new_state, brightness, brightness_pg):





        #pos =(pos*np.array(63)).astype(int)
        if new_state != self.old_state:

            self.old_state = new_state

            light_dir = np.array(new_state.light_dir) * 2 - 1
            camera_dir = np.array(new_state.camera_dir) * 2 - 1
            raw_location = np.array(new_state.location) * 2 - 1

            #camera_dir = np.array(new_state.lo) * 2 - 1

            input_info = aux_info.InputInfo()


            if new_state.lightfield_type == 0:
                input_info.light_dir_x = light_dir[0]
                input_info.light_dir_y = light_dir[1]

                import utils
                location = self.model.generate_locations()
                #location = utils.tensor.fmod1_1(location+.75)
            elif new_state.lightfield_type == 1:


                input_info.light_dir_x, input_info.light_dir_y = self.model.generate_light()
                location = self.model.generate_uniform_locations(raw_location[0],raw_location[1])



            input_info.camera_dir_x = camera_dir[0]
            input_info.camera_dir_y = camera_dir[1]

            input_info.mipmap_level_id =  new_state.mipmap_level_id




            if True:
                mimpap_type = new_state.mimpap_type
                do_blur = False



                input, mipmap_level_id = self.model.generate_input(input_info, use_repeat=True)

                if mimpap_type == 2:
                    mimpap_type = 0

                    blur = 2**float(mipmap_level_id[0,0,0,0].data.cpu())
                    mipmap_level_id = mipmap_level_id*0
                    do_blur = True

                result, eval_output = self.model.evaluate(input, location, level_id=mipmap_level_id, mimpap_type=mimpap_type,
                                         camera_dir=list(camera_dir))

                self.data_prob = eval_output.probability

                if do_blur:
                    pass
                    import utils
                    result = utils.tensor.blur(blur*.5*.75, result, True)

                if self.old_state.neural_offset_type == 1:
                    result = eval_output.neural_offset +.5
                    result = result.permute(0, 3, 1, 2)


                    result = eval_output.neural_offset_actual*4 +.5




                elif self.old_state.neural_offset_type == 2:
                    result = eval_output.shadow_neural_offset +.5

                    result = result.permute(0, 3, 1, 2)
                elif self.old_state.neural_offset_type == 3:
                    result = eval_output.neural_offset_2 +.5

                    result = result.permute(0, 3, 1, 2)

            zero_ch = result.shape[1]
            result = result.repeat([1, 3, 1, 1])
            result = result[:, :3, :, :]
            result[:,zero_ch:,:,:]=0
            result = result.data.cpu().numpy()[0,:,:,:].transpose([1,2,0])
            #result = self.data

            self.data = result

        if self.data is not None:
            result = self.data * brightness
        else:
            result = None


        prob_result = self.data_prob
        if prob_result is not None:
            prob_result = prob_result * brightness_pg



        #result = self.data[0,0,:32,:23]
        #result = result.cpu().numpy()#.transpose([1,2,0])
        return result, prob_result

class DatasetViewer:
    def __init__(self, brdf_model):

        self.patch_size = [32, 32]

        self.brdf_model = brdf_model

        self.data = None

        return

        if self.brdf_model.opt.namein is not None:
            self.data = np.load(self.brdf_model.opt.namein)[:,:,:,:,0]
        else:
            self.data = None

    def eval(self, x, y, brightness):



        pos = np.array((x,y))

        pos =(pos*np.array(63)).astype(int)
        print(pos)

        result = self.data[ :,:,pos[0], pos[1]]

        result = self.brdf_model.dataset.convert_back(result)

        result = np.clip(result, 0,1)
        result = pow(result, 1./2.2)

        #result = self.data[0,0,:32,:23]
        #result = result.cpu().numpy()#.transpose([1,2,0])
        return result




class Crosshair(tk.Canvas):
    def __init__(self, master, size, default_selected, callback=lambda:None, circle=False, background=None, zoom=1):
        self.f = callback

        self.circle = circle

        self.selector_size = np.array(size)

        default_selected = np.array(default_selected) #* self.selector_size

        self.selected =default_selected


        self.zoom = zoom

        if background is not None:
            background = background.clip(0,1)*255

            background = np.repeat(background, self.zoom, axis=0)
            background = np.repeat(background, self.zoom, axis=1)

            self.background = ImageTk.PhotoImage(
                image=Image.frombytes('RGB', (background.shape[1], background.shape[0]), background.astype('b').tostring()))

        else:
            self.background = None



        super().__init__(master, width=self.selector_size[0], height=self.selector_size[1], bg="#9999ff")

        self.bind("<Button-1>", self.callback)
        self.bind("<B1-Motion>", self.callback)

        self.callback2(default_selected)

    def callback(self, event):
        selected = np.array([event.x, event.y])/self.selector_size
        self.callback2(selected)
        self.f()

    def add_value(self, pos):
        selected = self.selected + pos/60
        self.callback2(selected)



    def callback2(self, selected):
        selected = np.array(selected)

        if self.circle:
            selected = selected *2.0 - 1.
            length = calc_length(selected)
            if length > .99:
                selected = selected/length

            selected = (selected + 1)*.5
        else:
            selected = np.clip(selected, 0, 1)

        if any(selected >= 1) or any(selected < 0):
            return

        self.delete("all")

        if self.background is not None:
            self.create_image(0, 0, image=self.background  , anchor=tk.NW)

        self.selected = selected
        if self.circle:
            self.create_oval(0,0,self.selector_size[0], self.selector_size[1])

        self.draw_crosshair(self.selected[0], self.selected[1])

    def draw_crosshair(self, x,y):
        self.create_line(0, y*self.selector_size[1], self.selector_size[0], y*self.selector_size[1])
        self.create_line(x*self.selector_size[0], 0, x*self.selector_size[0], self.selector_size[1])


class Viewer( tk.Canvas):
    def __init__(self, master=None,  size=None, zoom=1):

        super().__init__(master=master, width=size[0], height=size[1], bg="#000000")

        self.size = size
        self.zoom = zoom
        self.data = None

    def set_data(self, data):
        self.data = data
        self.update_zoom(self.zoom)

    def update_zoom(self, zoom):
        self.zoom = zoom

        self.photo = None
        if self.data is not None:
            new_view = self.data
            new_view = utils.tensor.to_output_format(new_view)
            new_view = new_view * 255
            new_view = np.repeat(new_view, self.zoom, axis=0)
            new_view = np.repeat(new_view, self.zoom, axis=1)
            self.photo = ImageTk.PhotoImage(
                image=Image.frombytes('RGB', (new_view.shape[1], new_view.shape[0]), new_view.astype('b').tostring()))

        self.redraw()



    def redraw(self):
        self.delete("all")
        if self.photo is not None:
            self.create_image(0, 0, image=self.photo, anchor=tk.NW)



class Joystick:
    def __init__(self):

        pygame.joystick.init()
        self.joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]

        for joystick in self.joysticks:
            joystick.init()
        pygame.init()


    def get_joystick_pos(self):
            j_a = np.array([0., 0.0])
            j_b = np.array([0., 0.0])

            pygame.event.pump()
            for joystick in self.joysticks:
                # axes = joystick.get_numaxes()
                # for idx in range(axes):
                #     print(joystick.get_axis(idx))
                cj_a = np.array([joystick.get_axis(0), joystick.get_axis(1)])
                cj_b = np.array([joystick.get_axis(3), joystick.get_axis(4)])

                if calc_length(cj_a) < .1:
                    cj_a = 0


                if calc_length(cj_b) < .1:
                    cj_b = 0

                j_a += cj_a
                j_b += cj_b


            return j_a, j_b


class State:
    def __init__(self):
        self.light_dir = None
        self.camera_dir = None
        self.location = None
        self.raw_buffer = None
        self.mipmap_level_id = None
        self.mimpap_type = None
        self.lightfield_type = None
        self.neural_offset_type = None


    def __eq__(self, other):
        if other == None:
            return False


        for var in vars(self):
            comp =  (getattr(self, var) != getattr(other, var))
            if not isinstance(comp, np.ndarray):
                if comp:
                    return False
            else:
                if comp.any():
                    return False

        return True


def main(model):


    brdf_model = None
    brdf_viewer = BRDFviewer(model, use_pathguiding=False)
    brdf_viewer_pg = BRDFviewer(model, use_pathguiding=True)
    #
    # ground_viewer = DatasetViewer(brdf_model)


    class MainApplication(tk.Frame):



        def draw_new_view(self, event=None):
            new_state = State()
            new_state.light_dir = self.light_selector.selected
            new_state.camera_dir = self.camera_selector.selected
            new_state.location = self.location_selector.selected
            new_state.raw_buffer = self.raw_buffer.get()
            new_state.mipmap_level_id = self.patch_level.get()
            new_state.mimpap_type = self.mimpap_type.get()
            new_state.lightfield_type = self.lightfield_type.get()
            new_state.neural_offset_type = self.neural_offset_type.get()

            if new_state.raw_buffer != -1:
                probability_view = None
                new_view = brdf_viewer.model.get_neural_texture_vis( new_state.neural_offset_type, new_state.raw_buffer, new_state.mipmap_level_id)
            else:
                new_view, probability_view = brdf_viewer.eval(new_state, self.brightness_b.get(), self.brightness_pg_b.get())
            self.viewer.set_data(new_view)
            self.viewer_pg.set_data(probability_view)

            # new_view_pg = brdf_viewer_pg.eval(new_state, self.brightness_b.get())
            # self.viewer_pg.set_data(new_view_pg)


            # new_view = ground_viewer.eval(x,y, self.brightness_b.get())
            # self.viewer_ground.set_data(new_view)

            #print(new_view.shape)


        def zoom_change(self, event=None):
            brdf_viewer.calculate(self.render_zoom.get())

            self.draw_new_view()


        def __init__(self, parent, *args, **kwargs):
            tk.Frame.__init__(self, parent, *args, **kwargs)

            self.js = Joystick()
            self.had_ground = False


            self.parent = parent

            self.left = tk.Frame(self)
            self.right = tk.Frame(self)

            self.brigtness = 1
            tk.Label(self.left, text='Brightness').grid()
            self.brightness_b = tk.Scale(self.left, from_=0, to=5, orient=tk.HORIZONTAL, length=500, resolution=.01, command=self.draw_new_view)
            self.brightness_b.grid()
            self.brightness_b.set(1)


            self.brigtness_pg = 1
            tk.Label(self.left, text='Brightness PG').grid()
            self.brightness_pg_b = tk.Scale(self.left, from_=0, to=5, orient=tk.HORIZONTAL, length=500, resolution=.01, command=self.draw_new_view)
            self.brightness_pg_b.grid()
            self.brightness_pg_b.set(.5)

            tk.Label(self.left, text='Texture vis').grid()
            self.raw_buffer = tk.Scale(self.left, from_=-1, to=7-3, orient=tk.HORIZONTAL, length=500, resolution=1, command=self.draw_new_view)
            self.raw_buffer.grid()

            tk.Label(self.left, text='Patch level').grid()
            self.patch_level = tk.Scale(self.left, from_=0, to=model.get_num_mipmap()-1, orient=tk.HORIZONTAL, length=500, resolution=.01, command=self.draw_new_view)
            self.patch_level.grid()


            tk.Label(self.left, text='Light Direction').grid()
            self.light_selector = Crosshair(self.left,  (256,256), (0.5,0.5), self.draw_new_view, True)
            self.light_selector.grid()


            tk.Label(self.left, text='Camera Direction').grid()
            #self.camera_selector = Crosshair(self.left, (256,256), (0.5,0.5), self.draw_new_view, True)
            self.camera_selector = Crosshair(self.left, (256, 256), (0.5, 0.5+0.286), self.draw_new_view, True)
            self.camera_selector.grid()

            tk.Label(self.left, text='Location').grid()
            self.location_selector = Crosshair(self.left, (256,256), (0.5,0.5), self.draw_new_view, False)
            self.location_selector.grid()

            self.camera_selector.grid_remove()
            self.camera_selector.grid()



            self.mimpap_type = tk.IntVar()
            self.mimpap_type.set(0)

            def init_radio(parent, name, var, types):

                radios = tk.Frame(parent)
                tk.Label(radios, text=name).grid()

                for name, val in (types):
                    tk.Radiobutton(radios,
                                   text=name,
                                   padx=20,
                                   variable=var,
                                   command=self.draw_new_view,
                                   value=val).grid()

                radios.grid()
                return radios




            self.lightfield_type = tk.IntVar()
            self.lightfield_type.set(0)

            init_radio(self.left, "Lightfield", self.lightfield_type, [
                ("Camera+Light", 0),
                ("Camera+Location", 1),
            ])


            self.neural_offset_type = tk.IntVar()
            self.neural_offset_type.set(0)

            init_radio(self.left, "Neural Offset", self.neural_offset_type, [
                ("Normal", 0),
                ("Neural Offset", 1),
                ("Shadow Neural Offset", 2),
                ("Neural Offset 2", 3),
            ])

            # self.raw_buffer = tk.IntVar()
            # self.raw_buffer.set(0)
            #
            # for idx, name in enumerate(["Raw Buffer", "Vis"]):
            #     tk.Radiobutton(root,
            #       text=name,
            #       variable=self.raw_buffer,
            #       value=idx, command=self.draw_new_view).pack()



            tk.Label(self.left, text='Iter {}'.format(model.iter_count)).grid()

            patch_size = np.array(model.resolution)
            zoom = 1

            self.viewer = Viewer(self.right, patch_size*zoom,zoom)
            self.viewer.grid()

            self.viewer_pg = Viewer(self.right, patch_size*zoom,zoom)
            self.viewer_pg.grid()


            if self.had_ground:
                self.viewer_ground = Viewer(self.right, patch_size*zoom,zoom)
                self.viewer_ground.grid()

            self.left.pack(side=tk.LEFT)
            self.right.pack(side=tk.LEFT)

            def updater():
                pos_a, pos_b = self.js.get_joystick_pos()

                self.light_selector.add_value(pos_a)
                self.camera_selector.add_value(pos_b)
                self.draw_new_view()

                self.after(16, updater)

            updater()




            # self.render_zoom = 1
            # self.render_zoom = tk.Scale(from_=1, to=8, orient=tk.HORIZONTAL, length=500, resolution=.01, command=self.zoom_change)
            # self.render_zoom.pack()
            # self.render_zoom.set(1)

            #self.viewer.set_data(model.get_neural_texture_vis())












    root = tk.Tk()
    MainApplication(root).pack(side="top", fill="both", expand=True)

    root.mainloop()

    #



if __name__ == "__main__":
    main(None)
