'''
Author Junbong Jang
Date: 2/8/2022

Refered to https://realpython.com/python-gui-tkinter/
This GUI is for creating the ground truth tracking points, similar to Polygonal Point Set Tracking

A user can click on the image to create the contour points
and those points are tracked across frames in the movie

'''
import tkinter as tk
from PIL import Image, ImageTk, ImageGrab
import numpy as np
from glob import glob
from os import path
import pyautogui  # essential for getting a correct mouse x,y coordinates
from datetime import datetime
import collections
import pylab
import os

def get_image_name(image_path, image_format):
    return image_path.split('\\')[-1].replace(image_format, '')


class LoadMovie:
    # refered to https://stackoverflow.com/questions/63128011/magnifying-glass-with-resizeable-image-for-tkinter
    def __init__(self, root_window, left_frame, right_frame, movie_path, save_path):
        self.save_path = save_path
        # make a save folder
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        # get images in the folder
        self.image_format = '.png'
        self.image_path_list = glob(f"{movie_path}/*{self.image_format}")
        self.movie_frame_num = 1


        # set up zoom canvas
        zoom_canvas_width = 300
        zoom_canvas_height = 200

        zoom_label = tk.Label(left_frame, text="Zoom In", width=20, height=2, fg="red", font=("Helvetica", 14))
        zoom_label.pack()

        self.zoom_canvas = tk.Canvas(left_frame, width=zoom_canvas_width, height=zoom_canvas_height, relief="solid", bd=2)
        self.zoom_canvas.pack()


        # set up label
        self.image_label = tk.Label(right_frame, text="Frame Number: 1", width=20, height=2, fg="red", font=("Helvetica", 14))
        self.image_label.pack(side='top')
        # color map for each loaded point
        self.color_map = pylab.get_cmap('gist_rainbow')

        # ------------------------------- set up main canvas -------------------------------
        orig_img = Image.open(self.image_path_list[0]).convert('RGB')
        img_width, img_height = orig_img.width, orig_img.height

        self.canvas_image_id = None
        canvas_frame = tk.Frame(right_frame, relief="solid", bd=2, width=500, height=400)
        canvas_frame.pack()

        self.canvas = tk.Canvas(canvas_frame, relief="solid", bd=2, width=700, height=400, scrollregion=(0,0,img_width,img_height))
        hbar = tk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        hbar.pack(side='bottom', fill=tk.X)
        hbar.config(command=self.canvas.xview)
        vbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        vbar.pack(side='right', fill=tk.Y)
        vbar.config(command=self.canvas.yview)
        self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.pack( expand=True, fill=tk.BOTH)

        self.display_image_on_canvas()
        self.canvas_oval_coords = np.zeros(shape=(img_height, img_width), dtype='uint16') # row, column
        self.tracking_id_from_oval_id = {}   # new oval id : unique tracking id

        # Bind user action to features
        # if scrolled, zoom in the image
        self.zoomcycle = 0
        self.zimg_id = None
        self.zoval_id = None
        self.prev_cursor_x = None
        self.prev_cursor_y = None
        self.canvas.bind("<MouseWheel>", self.zoomer)
        self.canvas.bind("<Motion>", self.crop)
        self.pix_tolerance = 20  # mouse cursor to the point proximity in pixels

        # If left-clicked, save x,y cursor coordinates
        self.canvas.bind('<ButtonRelease-1>', self.draw_circle)

        # if right-clicked, remove previously clicked coordinate
        # self.canvas.bind('<Button-3>', self.remove_circle)

        # if left-clicked moving, a closest tracking point is dragged
        self.canvas.bind('<B1-Motion>', self.drag_tracking_point)

        # ---------- Setup Label ------------
        self.tracking_label = tk.Label(right_frame, text=f"", width=40, height=2, fg="red", font=("Helvetica", 14))
        self.tracking_label.pack()
        self.update_tracking_label()
        self.load_tracking_points()

        # ---------- Setup buttons ----------
        paned_window_buttons = tk.PanedWindow(right_frame, relief="raised", bd=2)
        paned_window_buttons.pack()

        prev_button = tk.Button(paned_window_buttons, text='Prev', width=25, command=self.get_prev_frame, font=("Helvetica", 12))
        save_button = tk.Button(paned_window_buttons, text='Save', width=25, command=self.save_tracking_points, font=("Helvetica", 12))
        next_button = tk.Button(paned_window_buttons, text='Next', width=25, command=self.get_next_frame, font=("Helvetica", 12))

        paned_window_buttons.add(prev_button)
        paned_window_buttons.add(save_button)
        paned_window_buttons.add(next_button)

        clear_button = tk.Button(right_frame, text='Clear', width=25, command=self.clear_ovals_in_canvas, font=("Helvetica", 12))
        clear_button.pack()

        root_window.bind('<Left>', self.get_5_prev_frame)
        root_window.bind('<Right>', self.get_5_next_frame)
        root_window.bind('<KeyPress>', self.onKeyPress)


    def onKeyPress(self, event):
        print('onKeyPress:', event.char)
        if event.char == ',':
            self.get_prev_frame()
        elif event.char == '.':
            self.get_next_frame()
        elif event.char == 'm':
            self.save_tracking_points()

    def get_canvas_as_image(self, cur_x, cur_y, tol_x, tol_y):
        x = self.canvas.winfo_rootx() + cur_x
        y = self.canvas.winfo_rooty() + cur_y
        self.prev_cursor_x = cur_x
        self.prev_cursor_y = cur_y

        return ImageGrab.grab().crop((x-tol_x, y-tol_y, x+tol_x, y+tol_y))

    def zoomer(self, event):
        if (event.delta > 0):
            if self.zoomcycle != 5: self.zoomcycle += 1
        elif (event.delta < 0):
            if self.zoomcycle != 0: self.zoomcycle -= 1
        self.crop_x_y(event.x, event.y)

    def crop(self, event):
        if event is not None:
            self.crop_x_y(event.x, event.y)

    def crop_x_y(self, x, y):
        if self.zimg_id: self.zoom_canvas.delete(self.zimg_id)
        if self.zoval_id: self.zoom_canvas.delete(self.zoval_id)
        if (self.zoomcycle) != 0:
            if self.zoomcycle == 1:
                tmp = self.get_canvas_as_image(x,y, 90, 67.5)
            if self.zoomcycle == 2:
                tmp = self.get_canvas_as_image(x,y, 60, 45)
            elif self.zoomcycle == 3:
                tmp = self.get_canvas_as_image(x,y, 45, 30)
            elif self.zoomcycle == 4:
                tmp = self.get_canvas_as_image(x,y, 30, 20)
            elif self.zoomcycle == 5:
                tmp = self.get_canvas_as_image(x,y, 15, 10)
            size = self.zoom_canvas.winfo_width(), self.zoom_canvas.winfo_height()

            self.zimg = ImageTk.PhotoImage(tmp.resize(size))
            self.zimg_id = self.zoom_canvas.create_image(0, 0, image=self.zimg, anchor='nw')
            self.zoval_id = self.zoom_canvas.create_oval(self.zoom_canvas.winfo_width()/2-5, self.zoom_canvas.winfo_height()/2-5,
                                         self.zoom_canvas.winfo_width()/2+5, self.zoom_canvas.winfo_height()/2+5, outline ="red", width=1)

    def get_x_y_from_event(self, event):
        canvas = event.widget
        x = canvas.canvasx(event.x)  # event.x
        y = canvas.canvasy(event.y)  # event.y

        return int(x), int(y)

    def draw_circle(self, event):
        x, y = self.get_x_y_from_event(event)

        closest_oval_location = self.get_closest_oval_location(x, y, self.pix_tolerance)

        if closest_oval_location is None:
            print('closest oval is not found')
            new_oval_id = self.draw_circle_x_y(x, y, event)
            self.tracking_id_from_oval_id[new_oval_id] = new_oval_id

    def draw_circle_x_y(self, x, y, event=None, zoomin=True, outline_color="blue"):
        print('draw_circle: {}, {}'.format(x, y))
        if self.canvas_oval_coords[y, x] == 0:
            x1 = x-5
            y1 = y-5
            x2 = x+5
            y2 = y+5
            oval_id = self.canvas.create_oval(x1, y1, x2, y2, outline=outline_color, width=2)
            self.canvas_oval_coords[y, x] = oval_id
            # update zoom canvas
            if zoomin:
                self.zoom_canvas.after(1, self.crop, event)

            return oval_id
        else:
            print('Circle is already drawn at {}, {}'.format(x, y))
            return self.canvas_oval_coords[y, x]  # prev oval_id

    def clip_val(self, val, max):
        if val < 0:
            val = 0
        if val > max:
            val = max
        return val

    def get_closest_oval_location(self, x, y, pix_tolerance):
        # from the current mouse cursor position
        neighboring_oval_locations = (self.canvas_oval_coords
                                      [self.clip_val(y - pix_tolerance, self.canvas_oval_coords.shape[0]):
                                       self.clip_val(y + pix_tolerance, self.canvas_oval_coords.shape[0]),
                                      self.clip_val(x - pix_tolerance, self.canvas_oval_coords.shape[1]):
                                      self.clip_val(x + pix_tolerance, self.canvas_oval_coords.shape[1])] > 0)
        neighboring_oval_locations = np.transpose(neighboring_oval_locations.nonzero())

        min_dist = (pix_tolerance ** 2) * 2 + 1
        closest_oval_location = None
        for a_oval_location in neighboring_oval_locations:
            new_dist = (a_oval_location[0] - pix_tolerance)**2 + (a_oval_location[1] - pix_tolerance)**2
            if min_dist > new_dist:
                closest_oval_location = a_oval_location
                min_dist = new_dist

        return closest_oval_location

    def remove_circle(self, event):
        x, y = self.get_x_y_from_event(event)

        self.remove_circle_x_y(x, y, event)

    def remove_circle_x_y(self, x, y, event=None):
        print('remove_circle_x_y: {}, {}'.format(x, y))
        closest_oval_location = self.get_closest_oval_location(x, y, self.pix_tolerance)

        if closest_oval_location is None:
            print('closest oval is not found')
            closest_oval_id = None

        else:
            pix_tolerance_y = self.pix_tolerance
            pix_tolerance_x = self.pix_tolerance
            if y < self.pix_tolerance:
                pix_tolerance_y = y
            if x < self.pix_tolerance:
                pix_tolerance_x = x

            closest_oval_id = self.canvas_oval_coords[y+closest_oval_location[0]-pix_tolerance_y,
                                                      x+closest_oval_location[1]-pix_tolerance_x ]

            # remove from the canvas
            self.canvas.delete(closest_oval_id)
            # remove from the record
            self.canvas_oval_coords[
                y + closest_oval_location[0] - pix_tolerance_y, x + closest_oval_location[1] - pix_tolerance_x] = 0
            # update zoom canvas
            self.zoom_canvas.after(1, self.crop, event)

        return closest_oval_id

    def clear_ovals_in_canvas(self):
        neighboring_oval_locations = np.transpose((self.canvas_oval_coords > 0).nonzero())

        for a_oval_location in neighboring_oval_locations:
            oval_id = self.canvas_oval_coords[a_oval_location[0], a_oval_location[1] ]
            # remove from the canvas
            self.canvas.delete(oval_id)
            # remove from the record
            self.canvas_oval_coords[ a_oval_location[0], a_oval_location[1]] = 0

    def canvas_oval_coords_to_string(self):
        '''
        Save the coordinates to the text file with the following format
        column_coord row_coord\n

        :return:
        '''
        a_string = ""
        oval_location_dict = {}
        oval_locations = (self.canvas_oval_coords > 0 ).nonzero()  # np.transpose
        rows = oval_locations[0]
        columns = oval_locations[1]

        unique_tracking_ids_visible_in_canvas = []  # for handling removed points
        for a_coordinate in zip(rows, columns):
            oval_id = self.canvas_oval_coords[a_coordinate]
            unique_tracking_id = self.tracking_id_from_oval_id[oval_id]
            unique_tracking_ids_visible_in_canvas.append(unique_tracking_id)
            oval_location_dict[unique_tracking_id] = a_coordinate

        # Handle the removed points
        for oval_id, unique_tracking_id in self.tracking_id_from_oval_id.items():
            if unique_tracking_id not in unique_tracking_ids_visible_in_canvas:
                oval_location_dict[unique_tracking_id] = (-1, -1)

        # sort dict by keys
        od = collections.OrderedDict(sorted(oval_location_dict.items()))

        for k, a_coordinate in od.items():
            # reorder row & column coordinate to x & y coordinate
            a_string += str(a_coordinate[1]) + ' ' + str(a_coordinate[0]) + '\n'

        return a_string

    def drag_tracking_point(self, event):
        x, y = self.get_x_y_from_event(event)
        prev_oval_id = self.remove_circle_x_y(x, y, event)
        if prev_oval_id is not None:
            unique_tracking_id = self.tracking_id_from_oval_id[prev_oval_id]
            del self.tracking_id_from_oval_id[prev_oval_id]

            new_oval_id = self.draw_circle_x_y(x, y, event)
            self.tracking_id_from_oval_id[new_oval_id] = unique_tracking_id


    def save_tracking_points(self):
        a_image_name = get_image_name(self.image_path_list[ self.movie_frame_num - 1 ], self.image_format )
        save_string = self.canvas_oval_coords_to_string()
        with open(f'{self.save_path}/{a_image_name}.txt', 'w') as f:
            f.write(save_string)

        self.update_tracking_label()

    def load_tracking_points(self):
        def _load_tracking_points(load_path):
            self.tracking_id_from_oval_id = {}
            with open(load_path, 'r') as f:
                canvas_oval_coords = f.readlines()

            MAX_NUM_POINTS = len(canvas_oval_coords)

            for a_index, canvas_oval_coord in enumerate(canvas_oval_coords):
                a_coord = canvas_oval_coord.split()
                a_color = np.array([self.color_map(1. * a_index / MAX_NUM_POINTS)])*255
                a_color = a_color.astype('uint8')
                a_color_hex = "#%02x%02x%02x" % (a_color[0,0], a_color[0,1], a_color[0,2])

                if int(a_coord[0]) == -1:  # removed coordinate
                    new_oval_id = self.draw_circle_x_y( 0, 0, zoomin=False)
                    self.remove_circle_x_y( 0, 0)
                    self.tracking_id_from_oval_id[new_oval_id] = new_oval_id
                else:
                    new_oval_id = self.draw_circle_x_y(int(a_coord[0]), int(a_coord[1]), zoomin=False, outline_color=a_color_hex)
                self.tracking_id_from_oval_id[new_oval_id] = new_oval_id

        a_image_name = get_image_name(self.image_path_list[ self.movie_frame_num - 1 ], self.image_format )
        load_path = f'{self.save_path}/{a_image_name}.txt'
        if path.exists(load_path):
            _load_tracking_points(load_path)
            self.update_tracking_label(saved=True)

        else:
            # in a new frame, load previous frame's GT points
            prev_movie_frame_num = self.movie_frame_num - 2
            while prev_movie_frame_num >= 0:
                a_image_name = get_image_name(self.image_path_list[prev_movie_frame_num], self.image_format)
                load_path = f'{self.save_path}/{a_image_name}.txt'
                if path.exists(load_path):
                    _load_tracking_points(load_path)
                    self.update_tracking_label(saved=False)
                    break
                else:
                    prev_movie_frame_num = prev_movie_frame_num - 1

    def display_image_on_canvas(self):
        if self.canvas_image_id is not None:
            self.canvas.delete(self.canvas_image_id)  # delete previous image
            self.clear_ovals_in_canvas()
        orig_img = Image.open(self.image_path_list[self.movie_frame_num - 1]).convert('RGB')
        self.canvas_img = ImageTk.PhotoImage(orig_img)
        self.canvas_image_id = self.canvas.create_image(0, 0, image=self.canvas_img, anchor="nw")

    # -----------------------------------------
    def get_prev_frame(self):
        if self.movie_frame_num > 1:
            self.movie_frame_num = self.movie_frame_num - 1
            self.update_new_frame()

    def get_5_prev_frame(self, event):
        if self.movie_frame_num < 2:
            print()
        else:
            self.save_tracking_points()
            if self.movie_frame_num > 5:
                # If the frame_num is not in the multiple of 5
                a_remainder = (self.movie_frame_num - 5) % 5
                if a_remainder == 0:
                    self.movie_frame_num = self.movie_frame_num - 5
                else:
                    self.movie_frame_num = self.movie_frame_num - a_remainder
            elif self.movie_frame_num <= 5:
                self.movie_frame_num = 1
            self.update_new_frame()

    def get_next_frame(self):
        if self.movie_frame_num < len(self.image_path_list):
            # self.save_tracking_points()
            self.movie_frame_num = self.movie_frame_num + 1
            self.update_new_frame()

    def get_5_next_frame(self, event):
        if self.movie_frame_num >= len(self.image_path_list)-5:
            print()
        else:
            self.save_tracking_points()
            if self.movie_frame_num < 5:
                self.movie_frame_num = 5
            else:
                # If the frame_num is not in the multiple of 5
                a_remainder = (self.movie_frame_num + 5) % 5
                self.movie_frame_num = self.movie_frame_num + 5 - a_remainder
            self.update_new_frame()

    # -----------------------------------------

    def update_new_frame(self):
        print(f'----------- {self.movie_frame_num} ------------')
        self.display_image_on_canvas()
        self.load_tracking_points()
        self.zoom_canvas.after(1, self.crop_x_y, self.prev_cursor_x, self.prev_cursor_y)
        self.update_frame_label()

    def update_frame_label(self):
        self.image_label.config(text=f'Frame Number: {self.movie_frame_num}')

    def update_tracking_label(self, saved=True):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")

        if saved:
            self.tracking_label.config(text=f'Tracking Points: { np.count_nonzero(self.canvas_oval_coords>0) }; '
                                        f'Saved time: {current_time}')
        else:
            self.tracking_label.config(text=f'Tracking Points: { np.count_nonzero(self.canvas_oval_coords>0) }; '
                                        f'Not Saved')


if __name__ == "__main__":
    root_window = tk.Tk()
    root_window.title('Tracking Points Label tool')
    root_window.geometry("1100x725")

    # set up label
    left_frame = tk.Frame(root_window, relief="solid", bd=2)
    left_frame.pack(side="left", fill="both", expand=True)

    # heading_label = tk.Label(left_frame, text="How to use: \n Scroll to zoom. \n Finish labeling one point \n before adding one more point", font=("Helvetica", 12))
    # heading_label.pack(side="left", expand=True)

    right_frame = tk.Frame(root_window, relief="solid", bd=2)
    right_frame.pack(side="right", fill="both", expand=True)

    # for Phase contrast
    # 040119_PtK1_S01_01_phase, 040119_PtK1_S01_01_phase_ROI2, 040119_PtK1_S01_01_phase_2_DMSO_nd_01, 040119_PtK1_S01_01_phase_3_DMSO_nd_03
    LoadMovie(root_window, left_frame, right_frame,
              movie_path="generated/Computer Vision/PC_live/040119_PtK1_S01_01_phase/contour_points_visualize/",
              save_path='generated/Computer Vision/PC_live/040119_PtK1_S01_01_phase/points/')

    # for SDC dataset
    # 1_050818_DMSO_09, 2_052818_S02_none_08, 3_120217_S02_CK689_50uM_08, 4_122217_S02_DMSO_04, 5_120217_S02_CK689_50uM_07, 6_052818_S02_none_02, 7_120217_S02_CK689_50uM_13, 8_TFM-08122012-5, 9_052818_S02_none_12
    # LoadMovie(root_window, right_frame,
    #           movie_path="generated/Computer Vision/HACKS_live/5_120217_S02_CK689_50uM_07/contour_points_visualize/",
    #           save_path='generated/Computer Vision/HACKS_live/5_120217_S02_CK689_50uM_07/points/')

    # for jellyfish
    # LoadMovie(root_window, right_frame,
    #           movie_path="generated/Computer Vision/Jellyfish/First/contour_points_visualize/",
    #           save_path='generated/Computer Vision/Jellyfish/First/points/')

    root_window.eval('tk::PlaceWindow . center')
    root_window.mainloop()