# coding:utf-8
import sys
import io
from picamera import PiCamera
from time import sleep
from PIL import Image

sys.path.append("../HDMI/")
import hdmi

def setCamera(w, h):
  try:
    camera = PiCamera()
  except:
    print('Cannot initialize any camera!')
    return None

  camera.resolution = (w, h)

  return camera
  

def capture2PIL(camera, stream, cap_size):
  camera.capture(stream, format='rgb', use_video_port=True)
  raw_img = stream.getvalue()
  pil_img = Image.frombytes('RGB', (cap_size[0], cap_size[1]), raw_img, 'raw')
  
  stream.seek(0)
  stream.truncate()

  return pil_img


class PiCameraOverlay:
  mOverlayLayer = 3
  mCurrentOverlayLayer = 3
  mOverlay = None
  mCamera = None

  def __init__(self, camera, overlayLayer=3):
    self.mCamera = camera
    self.mOverlayLayer = overlayLayer
    self.mCurrentOverlayLayer = overlayLayer

  # https://github.com/waveform80/picamera/issues/448
  def OnOverlayUpdated(self, anOverlayImg, format='rgb', fullscreen=True, window=None):
    theTmpOverlay = self.mCamera.add_overlay(anOverlayImg.tobytes(), format=format, size=anOverlayImg.size, layer=self.mCurrentOverlayLayer, fullscreen=fullscreen, window=window)
    
    self.mCurrentOverlayLayer = self.mCurrentOverlayLayer + 1
    if self.mCurrentOverlayLayer > (self.mOverlayLayer + 1):
      self.mCurrentOverlayLayer = self.mOverlayLayer
    
    if self.mOverlay != None:
      self.mCamera.remove_overlay(self.mOverlay)

    self.mOverlay = theTmpOverlay

  def RemoveOverlay(self):
    if self.mOverlay != None:
      self.mCamera.remove_overlay(self.mOverlay)
      self.mOverlay=None


def main():
  CAP_HEIGHT = 256
  CAP_WIDTH  = 256

  camera = setCamera(CAP_WIDTH, CAP_HEIGHT)
  stream = io.BytesIO()

  # camera.start_preview()
  # sleep(10)

  while True:
    pil_img = capture2PIL(camera, stream, (CAP_WIDTH, CAP_HEIGHT)).convert('L')

    pil_img = hdmi.addBackImage(pil_img, *hdmi.getResolution())
    hdmi.addText(pil_img, *(300,   0), "camera")

    hdmi.printImg(pil_img, *hdmi.getResolution(), hdmi.PUT)

if __name__ == '__main__':
  main()
