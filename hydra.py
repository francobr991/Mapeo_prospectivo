from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string 
import numpy as np
import pandas as pd

from umap import UMAP


from plotly.subplots import make_subplots



import gdal

from itertools import product

from scipy.interpolate import LinearNDInterpolator


from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder

from seaborn import countplot,scatterplot,set_style
import matplotlib.pyplot as plt




def tokenize(text,idioma="spanish"):    
    '''
    Dado una cadena de caracteres, elimina símbolos especiales, stopwords y  devuelve una lista de tokens

    Parameters
    ----------
    text : string
        cadena de texto a tokenizar.
    idioma : string, optional
        idioma del stopword. The default is "spanish".

    Returns
    -------
    list
        lista de tokens

    '''
    #extrae las stop words segun el idioma
    stop=stopwords.words(idioma)    
    
    #convierte todo a minuscula
    data=text.lower()
    
    #separa palabras en tokens
    data = word_tokenize(data)     
    
    #reemplaza simbolos especiales por un punto ""
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    data = [re_punc.sub('', w) for w in data]
    
    #elimina palabras vacías
    data = [s for s in data if len(s) > 0]

    #devuelve las palabras que no estan en stop
    return [x for x in data if x not in stop]


def get_vector(palabra, model, return_zero=False):

   #dada una frase y un embedding, devuelve el vector
    epsilon = 1.e-10

    unk_idx = model.dictionary['unk']
    idx = model.dictionary.get(palabra, unk_idx)
    wv = model.word_vectors[idx].copy()

    if return_zero and palabra not in model.dictionary:
        n_comp = model.word_vectors.shape[1]
        wv = np.zeros(n_comp) + epsilon

    return wv

def mean_embeddings(data_frame,model,x="x",y="y",z="z",clase="clase",formacion="formacion"):
    
    DF=data_frame.copy()
    
    #elimina datos coordenados duplicados
    DF.drop_duplicates(subset=[x,y,z],inplace=True)
    
    #obtiene los tokens de cada descripcion
    DF['tokens'] = DF[formacion].apply(lambda x: tokenize(x,"english"))
    
    #guarda la longitud de cada cadena de palabras
    DF['longitud_descripciones'] = DF['tokens'].apply(lambda x: len(x))
    
    #transforma los tokens en vectores
    DF['vectors'] = DF['tokens'].apply(lambda x: np.asarray([get_vector(n, model) for n in x]))
    
    #mean obtiene el promedio de todas las palabras que componen la frase
    DF['mean'] = DF['vectors'].apply(lambda x: np.mean(x[~np.all(x == 1.e-10, axis=1)], axis=0))

    DF.reset_index(drop=True,inplace=True)    
    return DF

def well_plotly(datos,x="x",y="y",z="z",clase="clase",z_scale=-1,text=None,save_html=False,titulo="Pozos"):
    '''
    Parameters
    ----------
    datos : Data Frame
        set de datos con coordenadas X, Y, Z, y clase
    x : string, optional
        nombre de la columna de coordenada X. The default is "x".
    y : string, optional
        nombre de la columna de coordenada Y. The default is "y".
    z : string, optional
        nombre de la columna de coordenada Z. The default is "z".
    clase : string, optional
        nombre de la columna de clase. The default is "clase".
    z_scale : int o float, optional
        escalar que multiplica al eje Z. The default is -1.
    save_html : bool, optional
        si es TRUE guarda la grafica como HTML. The default is False.
    titulo : string, optional
        titulo de la grafica, y nombre con el que se guarda. The default is "Pozos".

    Returns
    -------
    None.

    '''
    
    num_class=dict(enumerate(np.sort(datos[clase].unique())))
    class_num={v: k for k, v in num_class.items()}   
    
    nombres=list(class_num.keys())
    
    fig = make_subplots(rows=1, cols=1,subplot_titles=([titulo]))
    for i in range(len(nombres)):
        sub_grup=datos[datos[clase]==nombres[i]]
        
    
        fig.add_scatter3d(  x=sub_grup[x], 
                            y=sub_grup[y], 
                            z=sub_grup[z]*z_scale,
                            mode='markers',
                            text=(sub_grup[text] if text!=None else ""),
                            name=nombres[i],
                            marker=dict(size=6,
                                        color=i,
                                        colorscale='Viridis',
                                        opacity=1,
                                        symbol="circle",
                                        line=dict(width=1,color='DarkSlateGrey')))
    
    fig.update_layout(height=900, width=900)
    if save_html==True:
        fig.write_html(f"{titulo}.html",)
    fig.show()

class plot_umap():    
    
    def __init__(self,DF,
                 vector="mean",
                 clases="clase",
                 n_components=3,
                 n_neighbors=50,
                 min_dist=0.0):
        '''
        Parameters
        ----------
        DF : DataFrame
            datos donde se almacenan los vectores que se van a reducir.
        vector : string, optional
            nombre de la columna donde se almacena el vector. The default is "mean".
        clases : string, optional
            nombre de la columna donde se almacena la clase. The default is "clase".
        n_components : int, optional
            dimension objetivo. The default is 3.
        n_neighbors : TYPE, optional
            parametro de UMAP. The default is 50.
        min_dist : TYPE, optional
            parametro de UMAP. The default is 0.0.

        Returns
        -------
        None.

        '''
        #se crea un modelo de UMAP para reducir la dimension de los datos
        self.modelo_UMAP=UMAP(n_components=n_components,
                         n_neighbors=n_neighbors,
                         min_dist=min_dist)
        
        #transforma los datos y los guarda en un DataFrame diferente
        self.data_umap=self.modelo_UMAP.fit_transform(np.asarray(list(DF[vector].copy().values)))
        self.plot_data=pd.DataFrame(self.data_umap,columns=["x","y","z"])
        
        #guarda las clases originales
        self.plot_data["clase"]=DF[clases].copy()
        
        #se crea un dicionario asignando un numero a cada etiqueta
        self.num_class=dict(enumerate(self.plot_data["clase"].unique()))
        #se invierte el diccionario
        self.class_num = {v: k for k, v in self.num_class.items()}


    def plot_plotly(self,titulo="Embedding",save_html=False):
        '''

        Parameters
        ----------
        titulo : string, optional
            titulo de la grafica y titulo del HTML si se guarda. The default is "Embedding".
        save_html : bool, optional
            si es TRUE guarda el HTML con el nombre del titulo. The default is False.

        Returns
        -------
        None.

        '''
        
        nombres=list(self.class_num.keys())
        
        fig = make_subplots(rows=1, cols=1,subplot_titles=([titulo]))
            
        for i in range(len(nombres)):
            sub_grup=self.plot_data[self.plot_data["clase"]==nombres[i]]


            fig.add_scatter3d(  x=sub_grup["x"], 
                                y=sub_grup["y"], 
                                z=sub_grup["z"],
                                mode='markers',
                                name=nombres[i],
                                marker=dict(size=6,
                                            color=i,
                                            colorscale='Viridis',
                                            opacity=1,
                                            symbol="circle",
                                            line=dict(width=1,color='DarkSlateGrey')))

        fig.update_layout(height=900, width=900)
        if save_html==True:
            fig.write_html(f"{titulo}.html",)
        fig.show()
 
def split_pozos(DF,x="x",y="y",test_size=0.1):
    
    #agrupa los datos del DataFrame por pozos, es decir que su coordenada X,Y sea igual
    grupo=DF.groupby([x,y])
    
    #selecciona una cantidad n de grupos, donde n es la cantidad de pozos * test_size
    sampling=np.random.randint(0,grupo.ngroups,size=int(grupo.ngroups*test_size))
    
    Test=DF[grupo.ngroup().isin(sampling)]
    Train=DF[~grupo.ngroup().isin(sampling)]
    
    return Train,Test
    

class get_model():

    def __init__(self,DF,DEM,intervalo_xy=10,franjas_z=10,x="x",y="y",z="z",test_size=0.1):
        
        self.DF=DF
        
        #se extrae la metadata del modelo digital de elevacion
        self.DEM=DEM
        self.DEM_meta = gdal.Open(self.DEM)
        self.DEM_geo = self.DEM_meta.GetGeoTransform()
        self.DEM_puntos = self.DEM_meta.GetRasterBand(1)  
        
        #se almacenan los nombres de los ejes y el test_size
        self.x=x
        self.y=y
        self.z=z
        self.test_size=test_size
        
        #se guardan los limites espaciales en cada eje
        self.x_min, self.x_max = self.DF[self.x].min(), self.DF[self.x].max()
        self.y_min, self.y_max = self.DF[self.y].min(), self.DF[self.y].max()
        self.z_min, self.z_max = self.DF[self.z].min(), self.DF[self.z].max()
        
        #se crea un vector de puntos en el eje X y en el eje Y, intervalo es la distancia que va a haber entre cada punto
        self.intervalo_xy=intervalo_xy
        self.points_int_x=np.arange(self.x_min, self.x_max, self.intervalo_xy)
        self.points_int_y=np.arange(self.y_min, self.y_max, self.intervalo_xy)
        
        #crea una malla entre los puntos del vector X con los puntos del vector Y
        self.xys_toint = list(product(self.points_int_x, self.points_int_y))
        
        #divide los pozos en entrenamiento y testeo
        self.pozos_train,self.pozos_test=split_pozos(self.DF,x=self.x,y=self.y,test_size=self.test_size)    
        
        self.calcular_franjas(franjas_z)
        
        
    def get_limits(self):
        print(f"x_min: {self.x_min}, x_max: {self.x_max}\ny_min: {self.y_min}, y_max: {self.y_max}\nz_min: {self.z_min}, z_max: {self.z_max}\n")
            
    
    def calcular_franjas(self,n_franjas):
        
        
        #el numero de franjas es la cantidad de tramos en el eje Z en el que se divide el modelo
        self.n_franjas=n_franjas
        
        
        #cada franja es un intervalo de profundidad, intervalo_z es el espesor de ese intervalo
        self.intervalo_z=self.z_max/self.n_franjas        
        #guarda los limites de las franjas
        self.franjas_z=[]  
        bandera=0
        for i in range(self.n_franjas):
            
            #calcula los limites de cada franja
            z = self.z_min+i
            
            #se seleccionan los registros de pozo que se encuentran en la franja, se verifica que el numero de pozos sea >2
            is_interpolable=self.pozos_train[(self.pozos_train[self.z] > z*self.intervalo_z) & (self.pozos_train[self.z] < (z+1)*self.intervalo_z)].copy()
            
            n_pozos=is_interpolable.groupby([self.x,self.y]).ngroups
            del(is_interpolable)
            
            if n_pozos>2:                
                self.franjas_z.append([z*self.intervalo_z*bandera,(z+1)*self.intervalo_z])
                bandera=1
            else:
                break
    
    def DF_from_inter(self,array):
        
        #dada una matriz proveniente de una interpolacion, la reestructura y la guarda en un data Frame        
        
        #pasa de tupla a matriz
        coordenadas=np.asarray(self.xys_toint)
        
        #crea un DataFrame vacio
        datos_total=pd.DataFrame(columns=["x","y","z","clase"])
        
        #itera por cada franja
        for i in range(len(array)):
            
            #guarda las coordenadas donde el dato no es nulo
            puntos=np.unique(np.where(array[i]>=0)[0])
            x_y=np.asarray(coordenadas[puntos])
            
            #guarda los datos en un DataFrame temporal
            df_temporal=pd.DataFrame(x_y,columns=["x","y"])
            df_temporal["z"]=np.mean(self.franjas_z[i])
            
            #guarda los valores interpolados no nulos y los agrega al DF
            clase=array[i][puntos]
            df_temporal["clase"]=list(clase)
            
            #concatena los DataFrame de cada franja
            datos_total=pd.concat([datos_total,df_temporal])
        
        datos_total.reset_index(drop=True,inplace=True)
        return datos_total.copy()    
    

    def interpolacion_lineal_basica(self,objetivo="clase",obj_type="vector"):
        '''

        Parameters
        ----------
        objetivo : string, optional
            nombre de la columna objetivo del DataFrame inicial. The default is "clase".
        obj_type : string, optional
            "vector" si objetivo es un vector, "clase" si el objetivo es string. The default is "vector".

        Returns
        -------
        None.

        '''
        
        if obj_type=="clase":
            #traduce la variable categorica a numeros
            num_class=dict(enumerate(self.DF[objetivo].unique()))
            class_num={v: k for k, v in num_class.items()}    
            
            #se codifican las clases
            local_encode = OneHotEncoder()        
            local_encode.fit(self.DF[objetivo].values.reshape([-1,1]))
            
            #se guardan las clases pertenecientes a objetivo en valor numerico y codificadas en onehotencode
            self.DF["num_"+objetivo]=self.DF[objetivo].copy().replace(class_num)        
            self.DF["encode_"+objetivo]=self.DF[objetivo].copy().apply(lambda x: local_encode.transform(np.array(x,ndmin=2)).toarray())
            
            
            self.pozos_train["num_"+objetivo]=self.pozos_train[objetivo].copy().replace(class_num)
            self.pozos_train["encode_"+objetivo]=self.pozos_train[objetivo].copy().apply(lambda x: local_encode.transform(np.array(x,ndmin=2)).toarray())
            
            self.pozos_test["num_"+objetivo]=self.pozos_test[objetivo].copy().replace(class_num)
            self.pozos_test["encode_"+objetivo]=self.pozos_test[objetivo].copy().apply(lambda x: local_encode.transform(np.array(x,ndmin=2)).toarray())

            #entrena un interpolador con todos los datos los datos
            self.interpolador=LinearNDInterpolator(self.pozos_train[[self.x, self.y,self.z]].values,np.array(self.pozos_train["encode_"+objetivo].tolist()))
            
            #convierte las coordenadas a matriz para poder agregar la profundidad
            coor=np.asarray(self.xys_toint)

            self.res_lineal_basica=[]       
            for i,j in tqdm(self.franjas_z):
                
                #profundidad a insertar en la matriz a interpolar
                insertar=np.ones(coor.shape[0]).reshape([-1,1])*np.mean([i,j])
                
                #interpola los datos de la zona
                self.res_lineal_basica.append(self.interpolador(np.hstack([coor,insertar])))                

                
            
            #pasa los datos de notacion matricial a un Data Frame
            self.res_lineal_basica=self.DF_from_inter(self.res_lineal_basica)
            
            #transforma los vectores en clases
            self.res_lineal_basica["clase"]=self.res_lineal_basica["clase"].apply(lambda x:local_encode.inverse_transform(x)[0][0])
            self.encode = local_encode
            
        elif obj_type=="vector":
            
            #entrena un interpolador con los datos dentro de la franja
            self.interpolador=LinearNDInterpolator(self.pozos_train[[self.x, self.y,self.z]].values,np.array(self.pozos_train[objetivo].tolist()))
            
            #convierte las coordenadas a matriz para poder agregar la profundidad
            coor=np.asarray(self.xys_toint)
            
            self.res_lineal_basica=[]       
            for i,j in tqdm(self.franjas_z):
                
                #profundidad a insertar en la matriz a interpolar
                insertar=np.ones(coor.shape[0]).reshape([-1,1])*np.mean([i,j])
                
                #interpola los datos de la zona
                self.res_lineal_basica.append(self.interpolador(np.hstack([coor,insertar])))
                
            
            #pasa los datos de notacion matricial a un Data Frame
            self.res_lineal_basica=self.DF_from_inter(self.res_lineal_basica)
        
        else:
            print("obj_type no valido, debe ser vector o clase")

    def show_map_data(self):

        coor_test = list(self.pozos_test.groupby(by=[self.x,self.y]).groups.keys())
        coor_test = np.asarray(coor_test)

        coor_train = list(self.pozos_train.groupby(by=[self.x,self.y]).groups.keys())
        coor_train = np.asarray(coor_train)

        set_style("darkgrid")
        scatterplot(x=coor_test[:,0],y=coor_test[:,1],color="red",label="Test")
        scatterplot(x=coor_train[:,0],y=coor_train[:,1],color="darkviolet",label="Train")

        plt.show()

    def show_class_data(self,clase):
        names = self.pozos_train[clase].unique()

        fig, ax = plt.subplots(1,2,figsize=(20,10))

        countplot(data = self.pozos_train, x=clase,ax=ax[0],order=names)
        ax[0].set_title("Train")
        ax[0].tick_params(axis='x', rotation=45)

        countplot(data = self.pozos_test, x=clase,ax=ax[1],order=names)
        ax[1].set_title("Test")
        ax[1].tick_params(axis='x', rotation=45)

        plt.show()
    


def calcular_altura(array,geo,raster,z_scale=1):
    
    #recalcula la altura de un punto segun la coordenada y el dem
    x,y,z=array
    
    #calcula la posicion del pixel presente en la coordenada mx,my
    px = int((x - geo[0])/geo[1])
    py = int((y - geo[3])/geo[5])    
    
    z=(z*z_scale)-raster.ReadAsArray(px, py, 1, 1)[0][0]
    
    return z

def plot_dem_well(DF_well, geo, raster, clase="clase",
                  x="x", y="y", z="z", titulo="surface_3D",
                  save_html=False, z_scale=-1,xy_muestreo=10,text=None):
    
    if text!=None:
        DF_well=DF_well[[x,y,z,clase,text]].copy()
    else:
        DF_well=DF_well[[x,y,z,clase]].copy()      
    
    
    #recalcula la altura de los pozos

    DF_well["n_z"]=DF_well[[x,y,z]].apply(lambda x:calcular_altura(x.values,geo,raster),axis=1)

    #extrae los nombres de las clases del dataset
    num_class=dict(enumerate(np.sort(DF_well[clase].unique())))
    class_num={v: k for k, v in num_class.items()}  
    nombres=list(class_num.keys())

    #figura sobre la cual se plotea 
    fig = make_subplots(rows=1, cols=1)

    # crea un scatter3d para cada clase 
    for i in range(len(nombres)):
        sub_grup=DF_well[DF_well[clase]==nombres[i]]

        fig.add_scatter3d(  x=sub_grup[x], 
                            y=sub_grup[y], 
                            z=sub_grup["n_z"]*z_scale,
                            mode='markers',
                            name=nombres[i],
                            text=(sub_grup[text] if text!=None else ""),
                            marker=dict(size=3,
                                        color=i,
                                        colorscale='Viridis',
                                        opacity=1,
                                        symbol="circle",
                                         )) 
    
    #datos para surface plot
    z_data = pd.DataFrame(raster.ReadAsArray())
    
    #cambia los datos negativos por el valor minimo mayor a cero 
    z_data[z_data<0]=z_data[z_data>0].min().min()
    
    #extrae el valor del eje mayor de la matriz
    eje_max=np.max(z_data.shape)
    
    #disminye la cantidad de puntos del DEM para liberar carga de trabajo
    data_z_plot=z_data[list(range(0,eje_max,xy_muestreo))].iloc[0:-1:10].values
    
    #calcula los ejes del surface plot, se basa en el origen del DEM (geo), tamaño del pixel (geo), numero de pixels (raster)
    #los ejes están trocados por la notacion interna de los formatos fijarse que data_z_plot.shape[1] se le asignó a X_axis
    x_axis=np.linspace(geo[0],geo[0]+(raster.XSize*geo[1]),data_z_plot.shape[1])
    y_axis=np.linspace(geo[3],geo[3]+(raster.YSize*geo[5]),data_z_plot.shape[0])

    fig.add_surface(z=data_z_plot,x=x_axis,y=y_axis,opacity=0.7,showscale=False)

    scene = {
                "xaxis": {"nticks": 20},
                "zaxis": {"nticks": 4},
                'camera_eye': {"x": 0, "y": -1, "z": 0.5},
                "aspectratio": {"x": 1, "y": 1, "z": 0.2}
            }

    fig.update_layout(title=titulo, autosize=True,
                      width=900, height=900,scene=scene)


    if save_html==True:
        fig.write_html(f"{titulo}.html",)
    fig.show()