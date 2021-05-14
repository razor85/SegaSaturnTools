# Copyright 2020-2021 Romulo Fernandes Machado Leitao <romulo@castorgroup.net>
# 
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import bpy
import bmesh
import copy
import ctypes
import math
import mathutils
import os
import pathlib
import gpu
import sys
import socket
from socket import ntohl
from socket import ntohs
from bpy import *
from math import *
from mathutils import Vector
from pathlib import Path

from bpy.app.handlers import persistent

bl_info = {
    "name": "Export Sega Saturn Model (.ssm)",
    "category": "Import-Export",
}

# Face flags
FLAG_DITHERING          = 1
FLAG_TRANSPARENCY       = 2
FLAG_IGNORE_FACE_SIZE   = 4
FLAG_GOURAUD            = 8
FLAG_SHADOW             = 16
EXPORTER_SETTINGS_NAME = 'saturn_ssm_exporter_settings' 

TEXTURE_WRITE_TEXTURE_STR = '__texture__'
TEXTURE_WRITE_ATLAS_STR   = '__atlas__'
TEXTURE_WRITE_NONE        = 'textureWriteNone'
TEXTURE_WRITE_TEX_ONLY    = TEXTURE_WRITE_TEXTURE_STR
TEXTURE_WRITE_ATLAS_ONLY  = TEXTURE_WRITE_ATLAS_STR
TEXTURE_WRITE_TEX_ATLAS   = '{}&&{}'.format(TEXTURE_WRITE_TEXTURE_STR,
                                            TEXTURE_WRITE_ATLAS_STR)


# Fields to save/load on exporter dialog
exportFields = ['filepath', 'animActions', 'checkForDuplicatedTextures',
                'useAO', 'texturesSizeByArea', 'minimumTextureSize',
                'outputTextureSize', 'saveLogFile', 'textureWriteOptions']

# Hold BMesh for each mesh.
globalMeshes = {}

def interpolateUv(a, b, amount):
  assert amount <= 1, 'Amount is in percentages'
  assert amount >= 0, 'Amount cant be negative'
  newUv = [ 0, 0 ]
  dist0 = b[0] - a[0]
  newUv[0] = a[0] + (amount * dist0)
  dist1 = b[1] - a[1]
  newUv[1] = a[1] + (amount * dist1)
  return newUv

def writeVector(filePtr, v):
  filePtr.write(fixedPoint(v.x))
  filePtr.write(fixedPoint(v.y))
  filePtr.write(fixedPoint(v.z))
      
def writeBlenderVector(filePtr, v):
  filePtr.write(fixedPoint(v.x))
  filePtr.write(fixedPoint(v.z))
  filePtr.write(fixedPoint(-v.y))

def closePowerOf2(value, maximum):
  newValue = 1
  while newValue <= value:
    newValue *= 2

  return min(newValue, maximum)

def getMeshForFaceIteration(obj):
 global globalMeshes
 if obj.name in globalMeshes:
   objMesh = globalMeshes[obj.name]
 else:
   objMesh = bmesh.new()
   objMesh.from_mesh(obj.data)

 objMesh.faces.ensure_lookup_table()
 return objMesh

def hasArmature(obj):
  return obj.parent is not None and obj.parent.type == 'ARMATURE'

def getVertexBoneIndex(obj, armature, vertex):
  groupIndex = vertex.groups[0].group
  boneName = obj.vertex_groups[groupIndex].name
  return armature.data.bones.find(boneName)

def getBoneIndices(obj):
  assert obj.parent.type == 'ARMATURE', 'Object parent must be an armature'
  armature = obj.parent
  boneIndices = []
  for vIndex, vertex in enumerate(obj.data.vertices):
    boneIndices.append(getVertexBoneIndex(obj, armature, vertex))
    
  return boneIndices

def removeGamma(pixel, gamma):
  newPixel = []
  for c in pixel:
    newPixel.append(pow(c, 1.0 / gamma))

  maxComponent = max(pixel[:3])
  if maxComponent > 1.0:
    diff = maxComponent - 1.0
    newPixel[0] -= diff
    newPixel[1] -= diff
    newPixel[2] -= diff

  for i in range(0, 3):
    assert newPixel[i] <= 1.0, 'Pixel is not in conversion interval'
    newPixel[i] = int(newPixel[i] * 255)

  return newPixel


def rgb555FromPixel(pixel):
  normalPixel = removeGamma(pixel, 1.8)
  r = ctypes.c_uint8(normalPixel[0])
  g = ctypes.c_uint8(normalPixel[1])
  b = ctypes.c_uint8(normalPixel[2])
  rgb555 = ctypes.c_uint16(((b.value >> 3) << 10)
           | ((g.value >> 3) << 5)
           | ((r.value >> 3) << 0)).value

  return rgb555


# image = [width, height, bpp, pixels]
def getPixel(uv, image):
  gammaValue = 1.8

  if uv[0] < -1:
    uv[0] = 1.0 - (uv[0] - int(uv[0]))
  elif uv[0] < 0:
    uv[0] = 1.0 + uv[0]
  elif uv[0] > 1:
    uv[0] = uv[0] - int(uv[0])

  if uv[1] < -1:
    uv[1] = 1.0 - (uv[1] - int(uv[1]))
  elif uv[1] < 0:
    uv[1] = 1.0 + uv[1]
  elif uv[1] > 1:
    uv[1] = uv[1] - int(uv[1])

  width, height, bpp, pixels = image
  x = int(round(width * uv[0]))
  y = int(round(height * uv[1]))
  x %= width
  y %= height
  baseIndex = int(y * width * bpp + x * bpp)
  assert baseIndex < len(pixels), 'Pixel index is out of bounds'

  newPixels = [ pow(pixel, gammaValue) for pixel in pixels[ baseIndex:baseIndex + bpp ] ]
  if len(newPixels) == 3:
    newPixels.append(1.0)

  return newPixels

def getColor(color):
  intColor = [(int)(color[0] * 255), (int)
              (color[1] * 255), (int)(color[2] * 255)]
  return (intColor[0] << 16) | (intColor[1] << 8) | intColor[2]


def fixedPoint(value):
  convertedValue = int(value * 65536.0)
  return ctypes.c_int32(ntohl(ctypes.c_uint32(convertedValue).value))

def applyOp(func, data):
  dataCopy = copy.deepcopy(data)
  for i in range(0, len(data)):
    dataCopy[i] = func(data[i])
    
  return dataCopy

def getEditBoneInverseEuler(bone):
  euler = bone.id_data.convert_space(pose_bone=bone,
                                     matrix=bone.matrix,
                                     from_space='POSE').to_euler()

  euler.rotate(mathutils.Euler([math.radians(-90), 0, 0]))
  return euler.to_matrix().inverted_safe().to_euler()
    
def getPoseBoneEuler(bone):
  matrix = bone.id_data.convert_space(pose_bone=bone,
                                      matrix=bone.matrix,
                                      from_space='POSE')

  rot90 = mathutils.Euler([math.radians(-90), 0, 0]).to_matrix().to_4x4()
  rotatedMatrix = rot90 @ matrix
  return rotatedMatrix.to_3x3().to_euler(), rotatedMatrix.to_translation()
  
def selectObj(obj):
  bpy.ops.object.select_all(action='DESELECT')
  obj.select_set(True)
  bpy.context.view_layer.objects.active = obj

def setArmaturePose(armature, pose):
  armature.data.pose_position = pose
  armature.data.update_tag()
  context.scene.frame_set(context.scene.frame_current)

class PoseBone:
  def __init__(self, name, head, tail, euler):
    self.name = name
    self.head = head
    self.tail = tail
    self.euler = euler
    self.numVertices = 0
    self.numFaces = 0

def getPoseData(obj):
  if not obj.parent or obj.parent.type != 'ARMATURE':
    return None
  
  armature = obj.parent
    
  # Toggle rest position
  oldPosition = armature.data.pose_position
  setArmaturePose(armature, 'REST')
  
  bones = armature.pose.bones.values()
  poseData = {}
  poseData['bones'] = []
  for bone in bones:
    invertedEuler = getEditBoneInverseEuler(bone)
    head = copy.deepcopy(bone.head)
    tail = copy.deepcopy(bone.tail)
    poseData['bones'].append(PoseBone(bone.name,
                                      head,
                                      tail,
                                      invertedEuler))
           
  setArmaturePose(armature, oldPosition)
    
  poseData['vertices'] = []
  objMesh = obj.to_mesh()
  for v in objMesh.vertices:
    vGroup = v.groups[0]
    assert vGroup.weight >= 1.0, 'Only 1.0 weighted vertices are allowed'

    boneName = obj.vertex_groups[vGroup.group].name
    boneIndex = armature.data.bones.find(boneName)
    poseData['vertices'].append(boneIndex)

    assert boneIndex < len(poseData['bones']), 'BoneIndex out of bounds'
    poseData['bones'][boneIndex].numVertices += 1

  # Calculate number of faces per bone
  objMesh = getMeshForFaceIteration(obj)
  for polyIndex, poly in enumerate(objMesh.faces):
    firstIndex = obj.data.loops[poly.loops[0].index].vertex_index
    firstVertex = obj.data.vertices[firstIndex]
    boneIndex = getVertexBoneIndex(obj, armature, firstVertex)

    assert boneIndex < len(poseData['bones']), 'BoneIndex out of bounds'
    poseData['bones'][boneIndex].numFaces += 1
    
  return poseData

def getAnimation(obj, actionName):
  if not obj.parent or obj.parent.type != 'ARMATURE' or actionName not in bpy.data.actions:
    return None
  
  armature = obj.parent
  action = bpy.data.actions[actionName]

  originalAction = armature.animation_data.action
  armature.animation_data.action = action
  frame_range = action.frame_range
  saveFrame = copy.deepcopy(bpy.context.scene.frame_current)
  
  frames = []
  for index in range(int(frame_range[0]), int(frame_range[1])):
    bpy.context.scene.frame_set(index)
    newFrame = []
    for boneIndex, bone in enumerate(armature.pose.bones):
      euler, position = getPoseBoneEuler(bone)
      newFrame.append([boneIndex, bone.name, position, euler])

    frames.append(copy.deepcopy(newFrame))
    
  armature.animation_data.action = originalAction
  bpy.context.scene.frame_set(saveFrame)
  return frames

def drawSuccessMessage(self, context):
  self.layout.label(text='Exporting successfull!')

class ExportSegaSaturnModel(bpy.types.Operator):
  """Export blender objects to Sega Saturn model"""
  bl_idname = "export.to_saturn"
  bl_label = "Export Saturn Model (.SSM)"
  bl_options = {'PRESET'}
  filepath: bpy.props.StringProperty(subtype='FILE_PATH')
  animActions: bpy.props.StringProperty(name="Actions", 
                                        description="Comma separated list of actions to export")
                                        
  checkForDuplicatedTextures: bpy.props.BoolProperty(name="Check for duplicated textures",
                                                     description="Check for duplicated textures",
                                                     default = True)
  useAO: bpy.props.BoolProperty(name="Use AO",
                                description="Use secondary texture/uv as AO",
                                default = False)

  texturesSizeByArea: bpy.props.BoolProperty(name="Use area to discover texture size",
                                             description="Use area to discover texture size",
                                             default = True)

  saveLogFile: bpy.props.BoolProperty(name="Save Log",
                                      description="Generate a log of the export",
                                      default = True)

  minimumTextureSize: bpy.props.IntProperty(name="Minimum Texture size",
                                            description="Minimum texture size",
                                            default = 8,
                                            max = 128,
                                            min = 2,
                                            step = 2)

  # Resolution of output textures.
  outputTextureSize: bpy.props.IntProperty(name="Texture Size (size x size)",
                                           description="Texture Size (size x size)",
                                           default = 16,
                                           max = 128,
                                           min = 2,
                                           step = 2)
  
  textureWriteOptionsEnum = [
    (TEXTURE_WRITE_ATLAS_ONLY,        'Atlas Only', ''),
    (  TEXTURE_WRITE_TEX_ONLY,      'Texture Only', ''),
    ( TEXTURE_WRITE_TEX_ATLAS, 'Texture and Atlas', ''),
    (      TEXTURE_WRITE_NONE,              'None', '')]

  textureWriteOptions: bpy.props.EnumProperty(items=textureWriteOptionsEnum,
                                              name="Texture Export",
                                              description="",
                                              default = 0)

  dontSaveSettings: bpy.props.BoolProperty(name="Dont save settings",
                                           description="Dont save those settings",
                                           default = False)

  # Keep textures by hash.
  modelTextures = {}
      
  # Keep texture bytes to make the atlas.
  modelTextureData = []

  # Index to each face texture file.
  faceTextures = []
  faceTextureSizes = []

  # Remap vertex position so we output all the vertices from the same
  # bone weight at once
  remappedVertexPositions = []
       
  largestArea = 0
  facesOffset = 0
  verticesOffset = 0
  exportedFaces = 0
  exportedVertices = 0
  
  def __init__(self):
    if EXPORTER_SETTINGS_NAME in bpy.context.scene:
      settings = bpy.context.scene[EXPORTER_SETTINGS_NAME]
      for field in exportFields:
        if field in settings:
          setattr(self, field, settings[field])

  
  @classmethod
  def poll(cls, context):
    return bpy.context.selected_objects is not None

  def findMinimumScale(self):
    assert len(bpy.context.selected_objects) > 0, 'No models selected'
    obj = bpy.context.selected_objects[0]
    objMesh = obj.data
    
    smallestValue = sys.float_info.max
    for f in objMesh.polygons:
      for v in f.vertices:
        smallestValue = min(objMesh.vertices[v].co[0], smallestValue)
        smallestValue = min(objMesh.vertices[v].co[1], smallestValue)
        smallestValue = min(objMesh.vertices[v].co[2], smallestValue)

    scale = 1.0
    while abs(smallestValue * scale) < 1.0:
      scale *= 10

    # print("Found scale of {}".format(scale))
    return scale
      
  def getMaximumArea(self):
    assert len(bpy.context.selected_objects) > 0, 'No models selected'
    obj = bpy.context.selected_objects[0]
    objMesh = getMeshForFaceIteration(obj)
    
    area = 0
    if objMesh.faces.layers.int.get("FaceFlags") is None:
      objMesh.faces.layers.int.new("FaceFlags")
      
    flagsLayer = objMesh.faces.layers.int.get("FaceFlags")
    for polygon in objMesh.faces:
      if (polygon[flagsLayer] & FLAG_IGNORE_FACE_SIZE) != 0:
        continue

      faceArea = polygon.calc_area()
      if faceArea > area:
        area = faceArea
      
    return area

  def writeFaces(self, filePtr, logFilePtr):
    self.facesOffset = filePtr.tell()
    if logFilePtr:
      logFilePtr.write("=============================================\n")
      logFilePtr.write("Starting writing faces at {} ({})\n".format(self.facesOffset,
                                                                    hex(self.facesOffset)))

    minimumScale = self.minimumScale
    totalIndex = 0
    vertexCount = 0
    faceIndex = 0

    assert len(bpy.context.selected_objects) > 0, 'No models selected'
    obj = bpy.context.selected_objects[0]
    objMesh = getMeshForFaceIteration(obj)
    if objMesh.faces.layers.int.get("FaceFlags") is None:
      objMesh.faces.layers.int.new("FaceFlags")
      
    flagsLayer = objMesh.faces.layers.int.get("FaceFlags")
    if logFilePtr:
      logFilePtr.write("Obj '{}' faces start at vertex index {}\n".format(obj.name, vertexCount))

    for polyIndex, poly in enumerate(objMesh.faces):
      indices = []
      if len(poly.loops) == 3:
        indices = [loop.index for loop in poly.loops[0:3]]
      elif len(poly.loops) == 4:
        indices = [loop.index for loop in poly.loops[0:4]]

      for loop_index in reversed(indices):
        index = ctypes.c_uint16(ntohs(obj.data.loops[loop_index].vertex_index + vertexCount))
        filePtr.write(index)

      if len(poly.loops) == 3:
        index = ctypes.c_uint16(ntohs(obj.data.loops[indices[0]].vertex_index + vertexCount))
        filePtr.write(index)

      filePtr.write(ctypes.c_uint16(ntohs(self.faceTextures[faceIndex])))
      filePtr.write(ctypes.c_uint8(self.faceTextureSizes[faceIndex]))
      filePtr.write(ctypes.c_uint8(poly[flagsLayer]))
      writeBlenderVector(filePtr, poly.normal)

      if logFilePtr:
        logIndices = [obj.data.loops[x].vertex_index + vertexCount for x in indices]
        if len(logIndices) == 3:
          logIndices.append(logIndices[0])

        logStr = '{} / IDX {}, {}, {}, {} / TEX {} / TEXSIZE {} / FLAG {} / N {}\n'
        logFilePtr.write(logStr.format(polyIndex,
                                       *logIndices,
                                       self.faceTextureSizes[faceIndex],
                                       self.faceTextureSizes[faceIndex],
                                       poly[flagsLayer],
                                       str(poly.normal)))

      faceIndex += 1
      totalIndex += 4
    
    # Add vertex count of active mesh to increase indices on next
    # object.
    vertexCount += len(obj.data.vertices)

  def writeVertices(self, filePtr, logFilePtr):
    self.verticesOffset = filePtr.tell()

    if logFilePtr:
      logFilePtr.write("=============================================\n")
      logFilePtr.write("Starting writing vertices at {} ({})".format(self.verticesOffset,
                                                                     hex(self.verticesOffset)))
    assert len(bpy.context.selected_objects) > 0, 'No models selected'
    obj = bpy.context.selected_objects[0]
    objMesh = obj.data
    
    minimumScale = self.minimumScale
    if logFilePtr:
      logFilePtr.write("{} vertices:\n".format(obj.name))

    for vIndex, v in enumerate(objMesh.vertices):
      copyV = v.co.copy()
      writeBlenderVector(filePtr, copyV * minimumScale)
      if logFilePtr:
        logFilePtr.write("{} = {} {} {}\n".format(vIndex, v.co[0], v.co[2], -v.co[1]))
  
        fV0 = fixedPoint( copyV.x)
        fV1 = fixedPoint( copyV.z)
        fV2 = fixedPoint(-copyV.y)
        logFilePtr.write("     {} {} {}\n".format(fV0.value, fV1.value, fV2.value))


  # Extract pixel values for the passed indices of a polygon.
  # indices = Indices of the vertices of the polygon.
  # uvLayer = UVW map layer to extract information from.
  # image = Image where to search the values from.
  # outTextureSize = Size of the size of output texture.
  def extractFaceTexturePixels(self, indices, uvLayer, image, outTextureSize):
    outputSize = outTextureSize
    outputWidth = outTextureSize
    outputHeight = outTextureSize
    uvs = [ uvLayer[ indices[ x ] ].uv for x in range(0, len(indices)) ]

    if len(indices) == 3:
      uv0 = uvs[ 2 ]
      uv1 = uvs[ 1 ]
      uv2 = uvs[ 0 ]
      uv3 = uvs[ 0 ]
    else:
      uv0 = uvs[ 3 ]
      uv1 = uvs[ 2 ]
      uv2 = uvs[ 1 ]
      uv3 = uvs[ 0 ]

    outputBytes = []
    for y in range(0, outputHeight):
      percentY = 1.0 - (y / (outputHeight - 1))
      for x in range(0, outputWidth):
        percentX = x / (outputWidth - 1)

        # Left vertex (in Y) interpolates between index 0 and 3
        # Right vertex (in Y) interpolates between index 1 and 2
        leftVertex = interpolateUv(uv0, uv3, percentY)
        rightVertex = interpolateUv(uv1, uv2, percentY)
        finalVertexPos = interpolateUv(leftVertex, rightVertex, percentX)
        pixelValue = getPixel(finalVertexPos, image) 
        outputBytes.extend(pixelValue)

    return outputBytes

              
  # polygonIndex = Index of polygon in model.
  # objData = All model data.
  # indices = Indices for each vertex/uv in face.
  # uvLayer = uvLayer to recover uv's from
  # image = [width, height, bpp, pixels]
  # texturesDir = Path() to textures directory
  def extractFaceTexture(self, polygonIndex, objData, indices, uvLayer, 
                         image, outTextureSize, texturesDir, aoLayer,
                         logFilePtr):

    outputSize = outTextureSize
    outputWidth = outTextureSize
    outputHeight = outTextureSize
    uvs = [ uvLayer[ indices[ x ] ].uv for x in range(0, len(indices)) ]
    outputBytes = self.extractFaceTexturePixels(indices, uvLayer, image, 
      outTextureSize)

    if aoLayer != None:
      assert len(aoLayer) == len(outputBytes), 'Image and AO must have the same size'
      for pixel in range(0, len(aoLayer)):
        outputBytes[pixel] *= aoLayer[pixel]

    hasTransparency = False
    reusedTexture = False

    # Do we have a texture with that info?
    texturesMatched = False
    if self.checkForDuplicatedTextures == True:
      textureHash = hash(frozenset(outputBytes))
      if textureHash in self.modelTextures:
        existingTexture = self.modelTextures[ textureHash ]
        texturesMatched = (outputBytes == existingTexture[0])
      
    self.faceTextureSizes.append(int(outputSize))
    if texturesMatched and self.checkForDuplicatedTextures == True:
      reusedTexture = True
      self.faceTextures.append( existingTexture[ 1 ] )

      if logFilePtr:
        logStr = "Duplicate of texture found on polygon {}, same as {}!\n"
        logFilePtr.write(logStr.format(polygonIndex, existingTexture[1]))

    else:
      if self.checkForDuplicatedTextures == True:
        self.modelTextures[ textureHash ] = [ outputBytes, polygonIndex ]

      newTextureData = []
      for i in range(0, outputHeight):
        lineIndex = (outputHeight - 1 - i) * outputWidth * 4
        lineRef = outputBytes[lineIndex:lineIndex + outputWidth * 4]
        newTextureData.extend(lineRef)
        
      self.modelTextureData.append( [ newTextureData[:], polygonIndex ] )
      self.faceTextures.append( polygonIndex )
      if self.textureWriteOptions.find(TEXTURE_WRITE_TEXTURE_STR) >= 0:
        newImageSavePath = Path(texturesDir / '{}.PNG'.format(polygonIndex))
        newImage = bpy.data.images.new(name='tmpImg_{}'.format(polygonIndex),
                                       width=outputWidth, 
                                       height=outputHeight, 
                                       alpha=False, 
                                       float_buffer=True)

        newImage.pixels = outputBytes
        newImage.filepath_raw = str(newImageSavePath)
        newImage.file_format = 'PNG'
        newImage.save()
        bpy.data.images.remove(newImage)

    return reusedTexture

  def getTextures(self, objData, matIndex):
    textures = []

    materials = objData.materials[matIndex]
    nodes = materials.node_tree.nodes
    textures.extend([n for n in nodes if n.type == 'TEX_IMAGE'])
    return textures

  def writeTextureAtlas(self, logFilePtr):
    filePath = Path(self.filepath)
    filePathTexturesDir = filePath.parents[0] / filePath.stem
    filePathTexturesDir.mkdir(parents=True, exist_ok=True)
    atlasPath = filePathTexturesDir / Path('ATLAS')

    with atlasPath.open("wb") as filePtr:
      for texture, index in self.modelTextureData:
        assert len(texture) % 4 == 0, "Texture bytes should be a multiple of 4."
        if logFilePtr:
          logFilePtr.write("Writing texture {} to atlas\n".format(index))

        for pixelIndex in range(0, int(len(texture) / 4)):
          realIndex = pixelIndex * 4
          pixel = texture[realIndex:realIndex + 4]
          rgb555 = ctypes.c_uint16(ntohs(0x8000 | rgb555FromPixel(pixel)))
          filePtr.write(rgb555)

  def extractModelTextures(self, logFilePtr):
    filePath = Path(self.filepath)
    filePathNoExt = filePath.parents[0] / filePath.stem
    filePathTexturesDir = filePath.parents[0] / filePath.stem
    if self.textureWriteOptions.find(TEXTURE_WRITE_TEXTURE_STR) >= 0:
      filePathTexturesDir.mkdir(parents=True, exist_ok=True)
    
    if logFilePtr: 
      logFilePtr.write("=============================================\n")
      logFilePtr.write("TextureDir: {}\n".format(filePathTexturesDir))

    assert len(bpy.context.selected_objects) > 0, 'No models selected'
    obj = bpy.context.selected_objects[0]
    objData = obj.data
    polygonIndex = 0

    # Extract pixels from UV.
    assert len(objData.uv_layers) > 0, 'Object must have a UV channel'
    uvLayer = objData.uv_layers[0].data
    aoLayer = None
    if self.useAO:
      assert len(objData.uv_layers) > 1, 'Object must have a secondary UV channel'
      aoLayer = objData.uv_layers[1].data
    
    for polyIndex, poly in enumerate(objData.polygons):
      matIndex = poly.material_index
      assert matIndex != None, 'Object must have a material'

      textures = self.getTextures(objData, matIndex)
      assert len(textures) > 0, 'Object material must have a texture'
      assert hasattr(textures[0], 'image'), 'Object material must have a texture image'

      texImgName = textures[0].image.name
      imgData = bpy.data.images[texImgName]
      width = imgData.size[0]
      height = imgData.size[1]
      bpp = imgData.channels
    
      outTextureSize = self.outputTextureSize
      if self.texturesSizeByArea:
        approximatedSize = (poly.area / self.largestArea) * self.outputTextureSize
        outTextureSize = closePowerOf2(approximatedSize, self.outputTextureSize)
        
      if outTextureSize < self.minimumTextureSize:
        outTextureSize = self.minimumTextureSize

      if logFilePtr: 
        logStr = "Face {} image '{}' ({}x{}@{} => {}x{})\n"
        logFilePtr.write(logStr.format(polyIndex,
                                       texImgName,
                                       width,
                                       height,
                                       bpp,
                                       self.outputTextureSize,
                                       self.outputTextureSize))

      # Extract pixels that way, otherwise it will be slow as hell 
      # because we would be accessing bpy_prop_array instead of a list.
      pixels = imgData.pixels[:]
      imgProperties = [width, height, bpp, pixels]

      assert bpp == 3 or bpp == 4, 'Only 24BPP or 32BPP images are supported'
      indices = None
      if len(poly.loop_indices) == 3:
        indices = poly.loop_indices[0:3]
      elif len(poly.loop_indices) == 4:
        indices = poly.loop_indices[0:4]

      aoTexture = None
      if aoLayer != None:
        aoImgName = textures[1].image.name
        aoData = bpy.data.images[aoImgName]
        assert imgData.channels == aoData.channels, 'Texture and AO must have the same bpp'
        aoProperties = [aoData.size[0], aoData.size[1], aoData.channels, aoData.pixels[:]]
        aoTexture = self.extractFaceTexturePixels(indices, aoLayer, 
                                                  aoProperties, 
                                                  outTextureSize)


      assert indices != None, "Polygon must have 3 or 4 vertices"
      reusedTexture = self.extractFaceTexture(polygonIndex, objData, 
                                              indices, uvLayer, 
                                              imgProperties, 
                                              outTextureSize,
                                              filePathTexturesDir, 
                                              aoTexture, logFilePtr)
        
      if not reusedTexture:
        polygonIndex += 1

    
  def writeAnimationHeader(self, filePtr, logFilePtr, poseData):
    bones = poseData['bones']
    vertices = poseData['vertices']
    
    # Count only bones that affect vertices
    validBones = [b for b in bones if b.numVertices > 0 and b.numFaces > 0]
    filePtr.write(ctypes.c_uint16(ntohs(len(validBones))))
    
    if logFilePtr:
      logFilePtr.write("=============================================\n")
      logFilePtr.write("Bones ({}) starting at {} ({})\n".format(len(validBones),
                                                                 filePtr.tell(),
                                                                 hex(filePtr.tell())))

    for bIndex, bone in enumerate(validBones):
      writeBlenderVector(filePtr, bone.head)
      sinEuler = applyOp(math.sin, bone.euler)
      cosEuler = applyOp(math.cos, bone.euler)
      writeVector(filePtr, sinEuler)
      writeVector(filePtr, cosEuler)
      filePtr.write(ctypes.c_uint32(ntohl(bone.numVertices)))
      filePtr.write(ctypes.c_uint32(ntohl(bone.numFaces)))

      if logFilePtr:
        logStr = "Bone {} '{}': START {} / EULER {} / VCOUNT: {} / FCOUNT: {}\n"
        logFilePtr.write(logStr.format(bIndex,
                                       bone.name,
                                       bone.head,
                                       bone.euler,
                                       bone.numVertices,
                                       bone.numFaces))

  def writeAnimation(self, filePtr, logFilePtr, animationData, poseData):
    if logFilePtr:
      logFilePtr.write("=============================================\n")
      logStr = "Animation starting at {} ({}) with {} frames\n"
      logFilePtr.write(logStr.format(filePtr.tell(),
                                     hex(filePtr.tell()),
                                     len(animationData)))

    poseBones = poseData['bones']
    frames = animationData
    for frameIndex, frame in enumerate(frames):
      if logFilePtr:
        logStr = "Frame {} starting at {} ({})\n"
        logFilePtr.write(logStr.format(frameIndex,
                                       filePtr.tell(),
                                       hex(filePtr.tell())))

      for bone in frame:
        index, name, position, euler = bone
        poseBone = poseBones[index]
        if poseBone.numVertices == 0 or poseBone.numFaces == 0:
          if logFilePtr:
            logFilePtr.write("  Skip Bone {} - {} - {} - {}\n".format(*bone))
            logFilePtr.write("  Matrix: {}\n".format(euler.to_matrix()))
        else:
          if logFilePtr:
            logFilePtr.write("  Bone {} - {} - {} - {}\n".format(*bone))
            logFilePtr.write("  Matrix: {}\n".format(euler.to_matrix()))

          writeVector(filePtr, position)
          writeVector(filePtr, applyOp(math.sin, euler))
          writeVector(filePtr, applyOp(math.cos, euler))

      if logFilePtr:
        logFilePtr.write("\n")

  # Sort object vertices and faces by bone index, that makes the transformation cache efficient
  def updateObjVerticesAndFaces(self, obj):
    if not obj.parent or obj.parent.type != 'ARMATURE':
      return

    selectObj(obj)
    boneIndices = getBoneIndices(obj)

    bpy.ops.object.mode_set(mode='EDIT')
    mesh = bmesh.from_edit_mesh(obj.data)
    sortedVertices = []
    for vertex in mesh.verts:
      sortedVertices.append([vertex.index, boneIndices[vertex.index]])
      
    sortedVertices.sort(key=lambda v: v[1])
    mesh.verts.ensure_lookup_table()

    vertexCounter = 0
    for vertex in sortedVertices:
      mesh.verts[vertex[0]].index = vertexCounter
      vertexCounter += 1
      
    mesh.verts.sort()
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Now sort faces with same criteria, but get the bone ids from
    # the 'normal' mesh data.
    mesh = getMeshForFaceIteration(obj)
    sortedFaces = []
    armature = obj.parent
    for polyIndex, poly in enumerate(mesh.faces):
      firstIndex = obj.data.loops[poly.loops[0].index].vertex_index
      firstVertex = obj.data.vertices[firstIndex]
      boneIndex = getVertexBoneIndex(obj, armature, firstVertex)
      sortedFaces.append([polyIndex, boneIndex])
    
    sortedFaces.sort(key=lambda v: v[1])

    # Apply sorted faces
    bpy.ops.object.mode_set(mode='EDIT')
    mesh = bmesh.from_edit_mesh(obj.data)
    mesh.verts.ensure_lookup_table()

    faceCounter = 0
    for faceIndex, boneIndex in sortedFaces:
      mesh.faces[faceIndex].index = faceCounter
      faceCounter += 1
    
    mesh.faces.sort()
    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

  def writeModelData(self, context, filePtr, logFilePtr):
    assert self.outputTextureSize % 2 == 0, "Texture size must be a multiple of 2"
    if EXPORTER_SETTINGS_NAME not in bpy.context.scene:
      bpy.context.scene[EXPORTER_SETTINGS_NAME] = {}
      
    settings = context.scene[EXPORTER_SETTINGS_NAME]
    if logFilePtr:
      logFilePtr.write("Model: {}\n".format(self.filepath))

    self.faceTextures = []
    self.faceTextureSizes = []
    self.modelTextures = {}
    self.modelTextureData = []
    self.minimumScale = 1.0
    self.exportedFaces = 0
    self.exportedVertices = 0
    self.largestArea = self.getMaximumArea()
    
    faceCount = 0
    vertexCount = 0

    assert len(bpy.context.selected_objects) > 0, 'No models selected'
    obj = bpy.context.selected_objects[0]
    self.updateObjVerticesAndFaces(obj)
    objMesh = obj.data
    objMesh.update()
    objMesh.calc_tangents()
    vertexCount += len(objMesh.vertices)
    for f in objMesh.polygons:
      assert len(f.vertices) == 3 or len(f.vertices) == 4, 'Only triangles and quads supported'
      faceCount += 1

    # Start writing to file
    filePtr.write(bytes([0x53, 0x41, 0x54, 0x2E]))
    filePtr.write(ctypes.c_uint16(ntohs(faceCount)))
    filePtr.write(ctypes.c_uint16(ntohs(vertexCount)))

    if logFilePtr:
      logFilePtr.write("FaceCount: {}\n".format(faceCount))
      logFilePtr.write("VertexCount: {}\n".format(vertexCount))

    # Offset where Faces and vertices start.
    offsetInFileForOffsets = filePtr.tell()
    filePtr.write(ctypes.c_uint16(ntohs(0))) 
    filePtr.write(ctypes.c_uint16(ntohs(0))) 

    # Extract textures and associate them with faces.
    self.extractModelTextures(logFilePtr)

    # Faces for frame
    self.writeFaces(filePtr, logFilePtr)

    # Vertices for frame
    self.writeVertices(filePtr, logFilePtr)

    filePtr.seek(offsetInFileForOffsets, 0)
    filePtr.write(ctypes.c_uint16(ntohs(self.facesOffset))) 
    filePtr.write(ctypes.c_uint16(ntohs(self.verticesOffset))) 

    # Write texture atlas.
    if self.textureWriteOptions.find(TEXTURE_WRITE_ATLAS_STR) >= 0:
      self.writeTextureAtlas(logFilePtr)

  def writeModelAnimationData(self, context, filePtr, logFilePtr):
    assert len(bpy.context.selected_objects) > 0, 'No models selected'
    obj = bpy.context.selected_objects[0]
    poseData = getPoseData(obj)
    if poseData and logFilePtr:
      logFilePtr.write("=============================================\n")
      logFilePtr.write('Model has pose data, writing\n')

    allActions = self.animActions.split(",")
    hasActions = False
    for animName in allActions:
      if animName in bpy.data.actions:
        hasActions = True
        break
      
    if hasActions and poseData:
      filePath = Path(self.filepath).with_suffix('.SSA')
      if logFilePtr:
        logFilePtr.write("Saving SSA to '{}'\n".format(self.filepath))
        
      allAnimationData = []
      totalNumFrames = 0
      for animName in allActions:
        animationData = getAnimation(obj, animName)
        allAnimationData.append(animationData)
        if animationData is not None:
          totalNumFrames += len(animationData)

      with filePath.open("wb") as filePtr:
        filePtr.write(bytes([0x53, 0x41, 0x54, 0x2E]))    
        self.writeAnimationHeader(filePtr, logFilePtr, poseData)
        filePtr.write(ctypes.c_uint16(ntohs(totalNumFrames)))

        for index, animName in enumerate(allActions):
          if logFilePtr:
            logFilePtr.write('Animation {}\n'.format(animName))

          animationData = allAnimationData[index]
          if animationData == None:
            continue
      
          self.writeAnimation(filePtr, logFilePtr, animationData, poseData)
    
  def execute(self, context):
    filePath = Path(self.filepath)
    print("Saving SSM to '{}'".format(self.filepath))

    if self.saveLogFile:
      logFilePath = filePath.with_suffix('.log')
      print("Saving SSM log to '{}'".format(logFilePath))

      with filePath.open("wb") as filePtr, logFilePath.open("w") as logFilePtr:
        self.writeModelData(context, filePtr, logFilePtr)
        self.writeModelAnimationData(context, filePtr, logFilePtr)

    else:
      with filePath.open("wb") as filePtr:
        self.writeModelData(context, filePtr, None)
        self.writeModelAnimationData(context, filePtr, None)
    

    # Save dialog settings
    settings = context.scene[EXPORTER_SETTINGS_NAME]
    if not self.dontSaveSettings:
      for field in exportFields:
        settings[field] = getattr(self, field)

    bpy.context.window_manager.popup_menu(drawSuccessMessage,
                                          title='Export finished',
                                          icon='INFO')
    return {'FINISHED'}

  def invoke(self, context, event):
      context.window_manager.fileselect_add(self)
      return {'RUNNING_MODAL'}


# Only needed if you want to add into a dynamic menu
def menu_func(self, context):
  self.layout.operator_context = 'INVOKE_DEFAULT'
  self.layout.operator(ExportSegaSaturnModel.bl_idname,
                       text="Export to Sega Saturn")

def setDithering(self, context):
  editObject = context.edit_object
  bm = globalMeshes.setdefault(editObject.name, 
                               bmesh.from_edit_mesh(editObject.data))

  for face in bm.faces:
    if face.select == False:
      continue

    layer = bm.faces.layers.int.get("FaceFlags")
    if bpy.context.window_manager.useDithering:
      face[layer] |= FLAG_DITHERING
    else:
      face[layer] &= ~FLAG_DITHERING

  return None

def setGouraud(self, context):
  editObject = context.edit_object
  bm = globalMeshes.setdefault(editObject.name, 
                               bmesh.from_edit_mesh(editObject.data))

  for face in bm.faces:
    if face.select == False:
      continue

    layer = bm.faces.layers.int.get("FaceFlags")
    if bpy.context.window_manager.useGouraud:
      face[layer] |= FLAG_GOURAUD
    else:
      face[layer] &= ~FLAG_GOURAUD

  return None

def setShadow(self, context):
  editObject = context.edit_object
  bm = globalMeshes.setdefault(editObject.name, 
                               bmesh.from_edit_mesh(editObject.data))

  for face in bm.faces:
    if face.select == False:
      continue

    layer = bm.faces.layers.int.get("FaceFlags")
    if bpy.context.window_manager.useShadow:
      face[layer] |= FLAG_SHADOW
    else:
      face[layer] &= ~FLAG_SHADOW

  return None

def setTransparency(self, context):
  editObject = context.edit_object
  bm = globalMeshes.setdefault(editObject.name, 
                               bmesh.from_edit_mesh(editObject.data))

  for face in bm.faces:
    if face.select == False:
      continue

    layer = bm.faces.layers.int.get("FaceFlags")
    if bpy.context.window_manager.useTransparency:
      face[layer] |= FLAG_TRANSPARENCY
    else:
      face[layer] &= ~FLAG_TRANSPARENCY

  return None

def setIgnoreFaceSize(self, context):
  editObject = context.edit_object
  bm = globalMeshes.setdefault(editObject.name, 
                               bmesh.from_edit_mesh(editObject.data))

  for face in bm.faces:
    if face.select == False:
      continue

    layer = bm.faces.layers.int.get("FaceFlags")
    if bpy.context.window_manager.ignoreFaceSize:
      face[layer] |= FLAG_IGNORE_FACE_SIZE
    else:
      face[layer] &= ~FLAG_IGNORE_FACE_SIZE

  return None

# Store intermediate values for face flags.
bpy.types.WindowManager.useDithering = bpy.props.BoolProperty(name="Use Dithering", 
                                                              update=setDithering)

bpy.types.WindowManager.useTransparency = bpy.props.BoolProperty(name="Use Transparency", 
                                                                 update=setTransparency)

bpy.types.WindowManager.ignoreFaceSize = bpy.props.BoolProperty(name="Ignore Face Size", 
                                                                update=setIgnoreFaceSize)

bpy.types.WindowManager.useGouraud = bpy.props.BoolProperty(name="Gouraud", 
                                                            update=setGouraud)

bpy.types.WindowManager.useShadow = bpy.props.BoolProperty(name="Use Shadow", 
                                                           update=setShadow)

# Update window manager values
def updateWMValues(bm):
  bm.faces.ensure_lookup_table()
  if bm.faces.layers.int.get("FaceFlags") is None:
    bm.faces.layers.int.new("FaceFlags")

  activeFaces = getActiveFaces(bm)
  if len(activeFaces) > 0:
    face = activeFaces[ 0 ]
    layer = bm.faces.layers.int.get("FaceFlags")
    bpy.context.window_manager.useDithering = ((face[layer] & FLAG_DITHERING) != 0)
    bpy.context.window_manager.useTransparency = ((face[layer] & FLAG_TRANSPARENCY) != 0)
    bpy.context.window_manager.ignoreFaceSize = ((face[layer] & FLAG_IGNORE_FACE_SIZE) != 0)
    bpy.context.window_manager.useGouraud = ((face[layer] & FLAG_GOURAUD) != 0)
    bpy.context.window_manager.useShadow = ((face[layer] & FLAG_SHADOW) != 0)

  return None

#scene update handler
@persistent
def editObjectChangeHandler(scene):
  selectedObjects = [x for x in scene.objects if x.select_get()]
  if len(selectedObjects) == 0:
    globalMeshes.clear()
    return None

  for obj in selectedObjects:
    # add one instance of edit bmesh to global dic
    if obj.mode == 'EDIT' and obj.type == 'MESH':
      bm = globalMeshes.setdefault(obj.name, bmesh.from_edit_mesh(obj.data))
      updateWMValues(bm)
      bmesh.update_edit_mesh(obj.data)

    # We left edit mode, clear mesh.
    elif obj.mode != 'EDIT' and obj.type == 'MESH':
      if obj.name in globalMeshes:
        globalMeshes[obj.name].free()
        globalMeshes.pop(obj.name)

  return None

def getActiveFaces(obj):
  faces = []
  for face in obj.faces:
    if face.select:
      faces.append(face)

  return faces

class ROMULO_PT_SaturnEditPanel(bpy.types.Panel):
  bl_idname = "ROMULO_PT_SaturnEditPanel"
  bl_label = "Sega Saturn"
  bl_region_type = "UI"
  bl_space_type = "VIEW_3D"

  @classmethod
  def poll(cls, context):
    # Only allow in edit mode for a selected mesh.
    return context.mode == "EDIT_MESH" and context.object is not None and context.object.type == "MESH"

  def draw(self, context):
    selectedObject = context.object
    bm = globalMeshes.setdefault(selectedObject.name, 
                                 bmesh.from_edit_mesh(selectedObject.data))

    activeFaces = getActiveFaces(bm)
    if len(activeFaces) > 1:
      self.layout.label(text="Multiple faces selected.")

    self.layout.prop(context.window_manager, "useDithering", text="Use Dithering")
    self.layout.prop(context.window_manager, "useTransparency", text="Use Transparency")
    self.layout.prop(context.window_manager, "ignoreFaceSize", text="Ignore Face Size")
    self.layout.prop(context.window_manager, "useGouraud", text="Gouraud")
    self.layout.prop(context.window_manager, "useShadow", text="Shadow")
    

def register():
  bpy.utils.register_class(ExportSegaSaturnModel)
  bpy.utils.register_class(ROMULO_PT_SaturnEditPanel)
  bpy.types.TOPBAR_MT_file_export.append(menu_func)

  # Face properties panel event handler.
  bpy.app.handlers.depsgraph_update_post.clear()
  bpy.app.handlers.depsgraph_update_post.append(editObjectChangeHandler)

def unregister():
  bpy.utils.unregister_class(ExportSegaSaturnModel)
  bpy.utils.unregister_class(ROMULO_PT_SaturnEditPanel)

  bpy.types.TOPBAR_MT_file_export.remove(menu_func)
  bpy.app.handlers.depsgraph_update_post.clear()


# This allows you to run the script directly from blenders text editor
# to test the addon without having to install it.
if __name__ == "__main__":
    register()

