import os
import numpy as np
import torch
import cv2
from torchvision.utils import save_image
import torch.nn.functional as F


mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

def normalize(img):
    return (img - 0.5) * 2.0

def unnormalize(img):
    return img * 0.5+0.5

def npremoveGamma(tensor):
    return np.power(tensor, 2.2)

#Add gamma to a vector
def npaddGamma(tensor):
    return np.power(tensor, 1.0/2.2)

def removeGamma(tensor):
    return torch.pow(tensor, 2.2)

#Add gamma to a vector
def addGamma(tensor):
    return torch.pow(tensor, 1.0/2.2)

def out_addgamma(outputs):
    outputs = output_process(outputs)
    outputs = unnormalize(outputs)
    normal = outputs[:, 0:3, :, :]
    diffuse = outputs[:, 3:6, :, :]
    roughness = outputs[:, 6:9, :, :]
    specular = outputs[:, 9:12, :, :]
    diffuse = addGamma(diffuse)
    specular = addGamma(specular)
    reconstructedOutputs = torch.cat(
        [normal, diffuse, roughness, specular], axis=1)
    return reconstructedOutputs


def makeNormal(partialOutputedNormals):
    tmpNormals = torch.ones(
        (partialOutputedNormals.shape[0], 1, partialOutputedNormals.shape[2], partialOutputedNormals.shape[3]),
        dtype=torch.float32).cuda()
    tensor = torch.cat([partialOutputedNormals, tmpNormals], dim=1)
    Length = torch.sqrt(torch.sum(torch.square(tensor), dim=1, keepdim=True))
    return torch.div(tensor, Length)

def output_process(outputs):
    if(outputs.shape[1]!=12):
        partialOutputedNormals = outputs[:, 0:2, :, :]
        outputedDiffuse = outputs[:, 2:5, :, :]
        outputedRoughness = outputs[:, 5, :, :]
        outputedSpecular = outputs[:, 6:9, :, :]

        normNormals = makeNormal(partialOutputedNormals)
        outputedRoughnessExpanded = outputedRoughness.unsqueeze(dim=1)
        reconstructedOutputs = torch.cat(
            [normNormals, outputedDiffuse, outputedRoughnessExpanded, outputedRoughnessExpanded, outputedRoughnessExpanded,
             outputedSpecular], dim=1)
        return reconstructedOutputs
    else:
        return outputs


def savetex(namepre, namelist, input):  # [batchsize,256,256,n]
    tex_list = torch.split(input, 3, dim=1)  # 4 * [batch, 256,256,3]
    for i, tex in enumerate(tex_list):
        t = torch.split(tex, 1, dim=0)
        for j, n in enumerate(namelist):
            name = n + '-' + namepre + '-{}.png'.format(i)
            image = torch.squeeze(t[j], 0)
            save_image(image, name)


def savetexto1(namepre, namelist, input, target):
    t = torch.split(input, 1, dim=0)  # [batch，12, 256,256]
    t2 = torch.split(target, 1, dim=0)
    for j, n in enumerate(namelist):
        tex_list = torch.split(t[j], 3, dim=1)
        tex_list2 = torch.split(t2[j], 3, dim=1)
        for i, image in enumerate(tex_list):# [1, 3, 256,256]
            if i == 0:
                images1 = image
            else:
                images1 = torch.cat([images1, image], 3)
        for i, image in enumerate(tex_list2):
            if i == 0:
                images2 = image
            else:
                images2 = torch.cat([images2, image], 3)
        images2 =  F.interpolate(images2, size=[images1.shape[2], images1.shape[3]], mode='bilinear', align_corners=True)
        images = torch.cat([images1, images2], 2)
        images = torch.squeeze(images)
        name = n + '-' + namepre + '.png'
        save_image(images, name)

def saverenderto1(namepre, namelist, input, target):# [batch, 256,256, 3*9]
    input = input.permute(0, 3, 1, 2)
    target = target.permute(0, 3, 1, 2)
    t = torch.split(input, 1, dim=0)
    t2 = torch.split(target, 1, dim=0)
    for j, n in enumerate(namelist):
        tex_list = torch.split(t[j], 3, dim=1)
        tex_list2 = torch.split(t2[j], 3, dim=1)
        for i, image in enumerate(tex_list):
            if i == 0:
                images1 = image
            else:
                images1 = torch.cat([images1, image], 3)
        for i, image in enumerate(tex_list2):
            if i == 0:
                images2 = image
            else:
                images2 = torch.cat([images2, image], 3)
        images = torch.cat([images1, images2], 2)
        images = torch.squeeze(images)
        name = n + '-' + namepre + '.png'
        save_image(images, name)

def saverender(namepre, namelist, input):
    render_list = torch.split(input, 1, dim=0)  # (9, 256, 256, 3)
    for i, renders in enumerate(render_list):
        renders = torch.squeeze(renders)
        renders = renders.permute(2, 0, 1)
        render = torch.split(renders, 3, dim=0)
        for j, t in enumerate(render):
            name = namelist[i] + '-' + namepre + '-{}.png'.format(j)
            image = torch.squeeze(t)
            save_image(image, name)


def saveimg(namepre, namelist, input):
    t = torch.split(input, 1, dim=0)
    for j, n in enumerate(namelist):
        name = n + '-' + namepre + '.png'
        image = torch.squeeze(t[j])
        save_image(image, name)