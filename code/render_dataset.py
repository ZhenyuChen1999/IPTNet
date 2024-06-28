import os
import random
import math
# MITSUBA_DIR="F:/Mitsuba_0.5.0/mitsuba.exe"

# Command to run renderer mitsuba.
# Node that the render process will fail if mitsuba 3 (as a python package) is installed.
MITSUBA_DIR="mitsuba"

# Dir for dataset to be saved.
SAVE_DIR="./dataset/"

# Dir for all render parameters to be saved. (The string) should not be modified.
# Parameters will be saved incrementally.
PARAM_DIR=SAVE_DIR+"params.txt"

# Dir for render objects. Any name combinations picked from OBJ_LIST & SUFFIX_LIST should exist within it.
OBJ_DIR="./renderObj/"

# Mitsuba scene description file. (The file) should not be modified.
XML_DIR="./scene.xml"

# The 'origin' of light source and the distance between the light source and 'origin'.
# The light source's location is randomly picked on an arc. 
LIGHT_BEGIN_COORD=[0.0,0.0,0.3]
LIGHT_DISTANCE_BEGIN=1.0
LIGHT_DISTANCE_END=2.0

# Range of light radiance or brightness.
LIGHT_RADIANCE_BEGIN=60.0
LIGHT_RADIANCE_END=120.0

# Range of $\sigma_t^\prime$ (reduced). The valid range of $\sigma_t^\prime$ is [0,+\inf].
SIGMA_T_BEGIN=25.0
SIGMA_T_END=300.0

# Range of $\Lambda^\prime$ (reduced). The valid range of $\Lambda^\prime$ is [0,1].
ALBEDO_BEGIN=0.3
ALBEDO_END=0.95

# Range of $g$. The valid range of $g$ is [-1,1].
# Note: parameter g has little effect to visual effect, yet increased g (abs(g) -> +1/-1) will increase noise greatly.
G_BEGIN=0.0
G_END=0.9

# Label strings of .xml file. (The strings) should not be modified.
XML_LIGHT_LOCATION="$lightLocation" # x="1" y="1" z="1"
XML_SIGMA_A="$sigmaA" # 1,1,1
XML_SIGMA_S="$sigmaS" # 1,1,1
XML_SIGMA_T="$sigmaT" # 1,1,1
XML_ALBEDO="$albedo" # 1,1,1
XML_RADIANCE="$radiance" # 160
XML_G_VALUE="$gValue" # 0.5
XML_OBJ="$obj" # ./obj/bunny.obj

# The number of generated images per combination of parameters.
LIGHT_NUM=10

# Index range for generated image groups.
PARAM_NUM_BEGIN=0
PARAM_NUM_END=600

# Name and suffix of .obj models within OBJ_DIR.
# Adjust local coordinates of .obj files before adding new models.
OBJ_LIST=["bunny","dragon","hex","monster","pegasus","buddha","bun","bust","cap","cone","cube","cup","cylinder","duck","kettle","lucy","soap","sphere","star","teapot"]
SUFFIX_LIST=["","_1","_2","_3","_4"]

def generateG():
    return str(random.uniform(G_BEGIN,G_END))

def generateObj():
    return OBJ_DIR+random.choice(OBJ_LIST)+random.choice(SUFFIX_LIST)+".obj"

def generateLocation(lightDistance):
    # only x and z
    xBias=random.uniform(-lightDistance,lightDistance)+LIGHT_BEGIN_COORD[0]
    yBias=0.0+LIGHT_BEGIN_COORD[1]
    zBias=math.sqrt(lightDistance**2-xBias**2)+LIGHT_BEGIN_COORD[2]
    return str.format("x=\"{x}\" y=\"{y}\" z=\"{z}\"",x=xBias,y=yBias,z=zBias)

def generateSigmaT():
    r,g,b=random.uniform(SIGMA_T_BEGIN,SIGMA_T_END),random.uniform(SIGMA_T_BEGIN,SIGMA_T_END),random.uniform(SIGMA_T_BEGIN,SIGMA_T_END)
    return str(r)+","+str(g)+","+str(b)

def generateAlbedo():
    r,g,b=random.uniform(ALBEDO_BEGIN,ALBEDO_END),random.uniform(ALBEDO_BEGIN,ALBEDO_END),random.uniform(ALBEDO_BEGIN,ALBEDO_END)
    return str(r)+","+str(g)+","+str(b)

def generateLightRadiance():
    return str(random.uniform(LIGHT_RADIANCE_BEGIN,LIGHT_RADIANCE_END))

def getReduced(sigmaTStr,albedoStr,gStr):
    sigmaTWords=sigmaTStr.split(",")
    albedoWords=albedoStr.split(",")
    
    sigmaTreal=[float(sigmaTWords[0]),float(sigmaTWords[1]),float(sigmaTWords[2])]
    albedoReal=[float(albedoWords[0]),float(albedoWords[1]),float(albedoWords[2])]
    gReal=float(gStr)
    
    sigmaTReduced=[.0,.0,.0]
    albedoReduced=[.0,.0,.0]
    sigmaSReal=[.0,.0,.0]
    sigmaSReduced=[.0,.0,.0]
    sigmaAReal=[.0,.0,.0]
    
    for i in range(3):
        sigmaSReal[i]=sigmaTreal[i]*albedoReal[i]
        sigmaAReal[i]=sigmaTreal[i]-sigmaSReal[i]
        sigmaSReduced[i]=(1.0-gReal)*sigmaSReal[i]
        sigmaTReduced[i]=sigmaSReduced[i]+sigmaAReal[i]
        albedoReduced[i]=sigmaSReduced[i]/sigmaTReduced[i]
        
    sigmaT=str(sigmaTReduced[0])+","+str(sigmaTReduced[1])+","+str(sigmaTReduced[2])
    albedo=str(albedoReduced[0])+","+str(albedoReduced[1])+","+str(albedoReduced[2])
    
    return sigmaT,albedo

def getReal(sigmaTStr,albedoStr,gStr):
    
    g=float(gStr)
    sigmaTWords=sigmaTStr.split(",")
    albedoWords=albedoStr.split(",")
    sigmaTreal=[.0,.0,.0]
    albedoReal=[.0,.0,.0]
    sigmaTReduced=[float(sigmaTWords[0]),float(sigmaTWords[1]),float(sigmaTWords[2])]
    albedoReduced=[float(albedoWords[0]),float(albedoWords[1]),float(albedoWords[2])]
    sigmaSReal=[.0,.0,.0]
    sigmaSReduced=[.0,.0,.0]
    sigmaAReal=[.0,.0,.0]
    # sigmaAReduced=[.0,.0,.0]
    
    for j in range(3):
        sigmaSReduced[j]=sigmaTReduced[j]*albedoReduced[j]
        sigmaSReal[j]=sigmaSReduced[j]/(1.0-g)
        sigmaAReal[j]=sigmaTReduced[j]-sigmaSReduced[j]
        sigmaTreal[j]=sigmaAReal[j]+sigmaSReal[j]
        albedoReal[j]=sigmaSReal[j]/sigmaTreal[j]
        
    sigmaT=str(sigmaTreal[0])+","+str(sigmaTreal[1])+","+str(sigmaTreal[2])
    albedo=str(albedoReal[0])+","+str(albedoReal[1])+","+str(albedoReal[2])
    
    return sigmaT,albedo

if __name__=="__main__":
    originXmlFile=open(XML_DIR)
    originXmlText=originXmlFile.read()

    for i in range(PARAM_NUM_BEGIN,PARAM_NUM_END):
        sigmaT=generateSigmaT()
        albedo=generateAlbedo()
        obj=generateObj()
        gValue=generateG()
        radiance=generateLightRadiance()
        lightDistance=random.uniform(LIGHT_DISTANCE_BEGIN,LIGHT_DISTANCE_END)
        paramTextFile=open(PARAM_DIR,"a+")
        
        sigmaTReal,albedoReal=getReal(sigmaT,albedo,gValue)
        
        # recording reduced sigma and albedo
        paramTextFile.write(str(i)+" "+obj+" "+sigmaT+" "+albedo+" "+gValue+"\n")
        paramTextFile.close()

        for _ in range(LIGHT_NUM):
            location=generateLocation(lightDistance)
            modifiedXmlText=originXmlText.replace(XML_LIGHT_LOCATION,location)
            modifiedXmlText=modifiedXmlText.replace(XML_SIGMA_T,sigmaTReal)
            modifiedXmlText=modifiedXmlText.replace(XML_ALBEDO,albedoReal)
            modifiedXmlText=modifiedXmlText.replace(XML_OBJ,obj)
            modifiedXmlText=modifiedXmlText.replace(XML_RADIANCE,radiance)
            modifiedXmlText=modifiedXmlText.replace(XML_G_VALUE,gValue)

            with open("./temp_with_g.xml","w") as f:
                f.write(modifiedXmlText)
            
            filename=SAVE_DIR+str(i)+"_"+str(_)+".exr"
            os.system(MITSUBA_DIR+" ./temp_with_g.xml"+" -o "+filename)

    
    paramTextFile.close()




