import tkinter
import tkinter.filedialog
import os
from PIL import ImageGrab
from time import sleep
import time
import orchttp
# import webview

root = tkinter.Tk()
root.geometry('100x50+200+800')
root.resizable(False, False)


class MyCapture:
    def __init__(self, png):
        # 用来记录鼠标按下位置
        self.X = tkinter.IntVar(value=0)
        self.Y = tkinter.IntVar(value=0)
        # 屏幕尺寸
        screenWidth = root.winfo_screenwidth()
        screenHeight = root.winfo_screenheight()
        # 创建顶级组建容器
        self.top = tkinter.Toplevel(root, width=screenWidth, height=screenHeight)
        # 不显示最大化最小化按钮
        self.top.overrideredirect(True)
        self.canvas = tkinter.Canvas(self.top, bg='white', width=screenWidth, height=screenHeight)
        # 显示全屏截图 在全屏截图上进行区域截图
        self.image = tkinter.PhotoImage(file=png)
        self.canvas.create_image(screenWidth / 2, screenHeight / 2, image=self.image)

        def onLeftButtonDown(event):
            self.X.set(event.x)
            self.Y.set(event.y)
            # 开始截图
            self.sel = True

        self.canvas.bind('<Button-1>', onLeftButtonDown)

        def onRightButtonDown(event):
            # 开始截图
            self.sel = False
            self.top.destroy()

        self.canvas.bind('<Button-3>', onRightButtonDown)

        def onLeftButtonMove(event):
            if not self.sel:
                return
            global lastDraw
            try:
                self.canvas.delete(lastDraw)
            except Exception as e:
                pass
            lastDraw = self.canvas.create_rectangle(self.X.get(), self.Y.get(), event.x, event.y, outline='black')

        self.canvas.bind('<B1-Motion>', onLeftButtonMove)

        def onLeftButtonUp(event):
            self.sel = False
            try:
                self.canvas.delete(lastDraw)
            except Exception as e:
                pass
            sleep(0.1)
            # 考虑反向截图
            left, right = sorted([self.X.get(), event.x])
            top, bottom = sorted([self.Y.get(), event.y])
            pic = ImageGrab.grab(bbox=(left + 1, top + 1, right, bottom))
            # 弹出保存对话框
            # fileName = tkinter.filedialog.asksaveasfilename(title='保存截图', filetypes=[('JPG files', '*.jpg')])
            global fileName
            fileName = './{:<.0f}.jpg'.format(time.time())
            if fileName:
                pic.save(fileName)

            # 关闭当前窗口
            self.top.destroy()
            orchttp.orc(fileName)
            # webview.load_url('https://www.baidu.com/')

        self.canvas.bind('<ButtonRelease-1>', onLeftButtonUp)
        self.canvas.pack(fill=tkinter.BOTH, expand=tkinter.YES)


def buttonCaptureClick():
    root.state('icon')
    sleep(0.1)
    filename = 'temp.png'
    im = ImageGrab.grab()
    im.save(filename)
    im.close()
    # 显示全屏截图
    w = MyCapture(filename)
    buttonCapture.wait_window(w.top)
    # 结束截图
    root.state('normal')
    os.remove(filename)


buttonCapture = tkinter.Button(root, text='截图', command=buttonCaptureClick)
buttonCapture.place(x=10, y=10, w=80, h=30)
root.mainloop()
# webview.create_window('sdadasd')
