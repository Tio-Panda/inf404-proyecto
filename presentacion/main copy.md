---
marp: true
paginate: true
theme: beam
header: Aplicación de deep learning para la mejora de calidad en imágenes ultrasonido usando beamforming
footer: Memoria para optar al título de ingeniero civil en informática
---

<!-- _class: title --->
<style scoped> h1{font-size:47px;}</style>

# Aplicación de deep learning para la mejora de calidad en imágenes de ultrasonido usando beamforming


<div style="position:relative; top: 150px; text-align: center">

<div style="text-align:center;">
<b>Memoria para optar al título de ingeniero civil en informática</b>
</div>
<div style="margin-top: 20px">

**Autor:** Sebastián Gutiérrez Milla

</div>

<div style="margin-top: 5px">

**Profesor Guía:** Julio Sotelo 
**Profesor Correferente:** Joaquín Mura

</div>

</div>

<div style="position:absolute;top:10px;left:10px; width:200px;">
<img src="image-1.png">
</div>

<div style="position:absolute;top:10px;right:10px; width:200px;">
<img src="image-2.png">
</div>


---

# Tabla de contenidos

<div class="beam-toc">
    <div class="beam-toc-item">
        <div class="beam-number"></div>
        <div class="beam-content">
            <h3 class="beam-section-title">Contexto</h3>
        </div>
    </div>
    
  <div class="beam-toc-item">
      <div class="beam-number"></div>
      <div class="beam-content">
          <h3 class="beam-section-title">Marco teórico</h3>
      </div>
  </div>
    
  <div class="beam-toc-item">
      <div class="beam-number"></div>
      <div class="beam-content">
          <h3 class="beam-section-title">Problema</h3>
      </div>
  </div>
  
  <div class="beam-toc-item">
      <div class="beam-number"></div>
      <div class="beam-content">
          <h3 class="beam-section-title">Solución</h3>
      </div>
  </div>

  <div class="beam-toc-item">
      <div class="beam-number"></div>
      <div class="beam-content">
          <h3 class="beam-section-title">Resultados</h3>
      </div>
  </div>
  
  <div class="beam-toc-item">
      <div class="beam-number"></div>
      <div class="beam-content">
          <h3 class="beam-section-title">Discusión y conclusiones</h3>
      </div>
  </div>
</div>

---
# Contexto - ¿Qué es el ultrasonido?

El ultrasonido es una modalidad no ionizante que usa pulsos acústicos y el tiempo de vuelo de sus ecos para producir imágenes en tiempo real. Origen: inspirado en el sonar del siglo XX (post-Titanic) y adaptado a la medicina tras la Segunda Guerra Mundial.

---
# Contexto - ¿Por qué es importante?

- **Seguridad**: Sin radiación ionizante; historial clínico muy favorable desde los 50s.

- **Portabilidad**: Dispositivos de mano/bolsillo con acceso masivo y uso en terreno

- **Costo**: Una de las modalidades más accesibles en compra, operación y mantención en comparación a sus contrapartes (resonancia magnética y tomografía computarizada).

---
# Contexto - PICMUS y CUBDL

**Plane-wave Imaging Challenge in Medical UltraSound (PICMUS)** es un desafío académico centrado en plane-wave imaging para ultrasonido. Este estandariza datos, protocolos y métricas para comparar objetivamente beamforming en plane-wave/ultrafast imaging.

**Challenge on Ultrasound Beamforming with Deep Learning (CUBDL)** es un desafío orientado a aplicar deep learning al beamforming de ultrasonido. Nació para explorar, validar y guiar el uso de deep learning que igualen o superen a los métodos clásicos.

---
# Contexto - Aplicaciones de deep learning al ultrasonido

**Ultrasound Beamforming using MobileNetV2 (Goudarzi et al., 2020)**: usan MobileNetV2 para estimar los pesos de MVB, alineando y demodulando las señales para mantener fidelidad física. Logran calidad similar a MVB (mejor que DAS) y reducen el tiempo de 4,05 a 0,67 min (aún no en tiempo real).

---

**Beamforming-integrated neural networks for ultrasound imaging (Xiao et al.,2025)**: integran una capa de sparse matrix beamforming (SMB) en una CNN (BINN), codificando DAS explícitamente. Alcanzan ~5 ms de inferencia y una mejora del 9,8 % en MSLE frente a una CNN estándar, viable para ultrafast imaging.

---
# Marco teórico - Transductor

Instrumento encargado de hacer el ultrasonido posible. Este emite ondas acústicas al tejido y es capaz de recibir los ecos para conformar la imagen final. En equipos modernos esto se logra mediante arreglos unidimensionales de elementos piezoeléctricos que pueden disponerse en configuraciones lineales, convexas o fásicas.

![center w:800 h:300](image.png)

---
# Marco teórico - Beamforming

El beamforming es la técnica que combina las señales de múltiples elementos de un transductor para “formar” y dirigir un haz hacia una región de interés, permitiendo extraer información de esa zona.

![center w:750 h:350](image-3.png)

---
# Marco teórico - Cálculos de tiempos de vuelo (ToF)

![center w:750 h:350](image-4.png)

---

$$\tau_{tx}(p, \alpha) = \frac{1}{c_0}(x\sin(\alpha) + z\cos(\alpha))$$

$$\tau_{rx}(p, x_c) = \frac{1}{c_0}\sqrt{z^2 + (x - x_1)^2}$$

$$\tau(p, \alpha, x_c) = \tau_{tx}(p, \alpha) + \tau_{rx}(p, x_c)$$

---
# Marco teórico - Delay-and-Sum (DAS)

Delay-and-Sum (DAS) es el beamformer lineal y digital más simple: para cada punto de imagen calcula los tiempos de vuelo de transmisión y recepción, alinea las señales de todos los elementos del transductor e inmediatamente las suma de forma coherente para estimar la amplitud en ese punto.

---

Sea $s_c(t)$ la señal del elemento $c$ en el instante $t$ y $N_c$ la cantidad de elementos del transductor.

$$y_{\textit{DAS}}(t) = \sum_{c=1}^{N_c} s_c(t)$$

---
# Marco teórico - Delay-Multiply-and-Sum (DMAS)

DMAS (Delay Multiply and Sum) es un beamformer no lineal que, tras alinear las señales como en DAS, multiplica combinatoriamente los canales y suma los términos resultantes.

---

Sea $s_c(t)$ la señal del elemento $c$ en el instante $t$, $N_c$ la cantidad de elementos del transductor y $sign$ la función signo.

$$\hat s_{ij}(t) = sign(s_i(t) s_j(t)) \times \sqrt{|s_i(t) s_j(t)|}$$

$$y_{DMAS}(t) = \sum_{i=1}^{N_c-1} \sum_{j=i+1}^{N_c} \hat s_{ij}(t)$$

---
# Marco teórico - Minimum Variance Beamforming (MVB)

Minimum Variance es un beamformer adaptativo que calcula pesos a partir de los datos para minimizar la potencia de salida con la restricción de mantener respuesta unitaria hacia el punto focal; así suprime interferencias y ruido, mejorando resolución y contraste frente a métodos no adaptativos como DAS.

---

Sea $s_c(t)$ la señal recibida en el elemento $c$ y $\tau_c(t)$ el delay para el instante $t$

$$x_c(t) ;=; s_c\big(t-\tau_m(t)\big),
\qquad \mathbf{x}(t)=\begin{bmatrix}x_0(t)&\cdots&x_{c-1}(t)\end{bmatrix}^{T}$$

$$y_{MVB}(t) = \textbf{w}^{H}[t] \textbf{x}[t]$$

$$\arg\min_{\mathbf{w}}\ \mathbf{w}^{H}\mathbf{R}_{xx}(t),\mathbf{w}
\quad \text{sujeto a}\quad
\mathbf{w}^{H}\mathbf{a}(t)=1;$$

donde ($\mathbf{R}_{xx}(t)=\mathbb{E}{\mathbf{x}(t)\mathbf{x}^{H}(t)}$) es la matriz de covarianza local, $\mathbf{a}$ el vector de apuntamiento el cual queda como $1$ gracias al prealineamiento de las señales y $H$ conjugada transpuesta.

---
# Marco teórico - Métricas de contraste

**CNR:** cuantifica el contraste entre dos regiones (p. ej., lesión y fondo) relativo al ruido total; mayor CNR ⇒ mejor separabilidad visual.
$$ \mathrm{CNR}=\frac{|\mu_i-\mu_o|}{\sqrt{\sigma_i^2+\sigma_o^2}} $$

**gCNR:** mide la separabilidad robusta al speckle usando la superposición de distribuciones normalizadas de intensidad; 0 = solapadas, 1 = perfectamente separables.
$$ \mathrm{gCNR}=1-\int_{-\infty}^{\infty}\min\big(p_i(x),p_o(x)\big) dx $$


---
# Marco teórico - Ultrafast Imaging

Ultrafast imaging es un modo de adquisición en ultrasonido que ilumina toda la región de interés con ondas planas (SPW) y realiza beamforming paralelo en recepción, permitiendo cientos o miles de fps en lugar del barrido línea por línea tradicional.

![center w:750 h:350](image-5.png)

---
# Marco teórico - Coherent Plane-Wave Compounding (CPWC)

Coherent Plane-Wave Compounding (CPWC) es la extensión natural de SPW para recuperar la calidad de la reconstrucción. El principio es emitir múltiples SPW con distintos ángulos de inclinación, hacer la reconstrucción para cada una de estas y luego sumarlas.

---
# Problema

En *ultrafast imaging* con SPW se pierde calidad (contraste/resolución) y CPWC la recupera a costa de mayor cómputo y menor *frame-rate*.

Se define el *trade-off* entre calidad de la imagen–velocidad de reconstrucción.

---
# Solución - Objetivo

Implementar y validar un método de deep learning para reconstruir desde una sola SPW con calidad comparable/superior a una reconstrucción utilizando CPWC.

---
# Solución - Comparación entre DAS, DMAS y MVB para encontrar un *ground truth*

Para comparar DAS, DMAS y MVB y elegir el *ground truth*, se reconstruyeron imágenes del reto PICMUS y se evaluaron con métricas de contraste (CNR y, como criterio principal, gCNR) ademas, se incluyeron además adquisiciones *in vivo* para una comparación visual.

---

![center w:950 h:450](image-8.png)

---

![center w:950 h:450](image-9.png)

---

![center w:950 h:450](image-10.png)

---

![center w:900 h:300](image-11.png)

---

![center w:950 h:450](image-6.png)

---

![center w:950 h:450](image-7.png)

---
# Solución - Arquitectura de la red

![center w:950 h:450](image-12.png)

---
# Solución - Dataset

- **Origen**: Combinación de datos públicos de **PICMUS** y **CUBDL** para formar el conjunto de entrenamiento/validación y evaluación.

- **Contenidos**: 561 adquisiciones CPWC, todas con transductor de 128 elementos; hay tres rangos de ángulos: 75, 73 y 31 SPW según la adquisición.

- **Entrada del modelo**: matriz **RF**, grid de imagen, posiciones de los elementos del transductor y parámetros (frecuencia de muestra ($f_s$), tiempo inicial ($t_0$), velocidad del sonido ($c$)). 

- **Ground truth:** Reconstrucción obtenida con MVB usando todos los ángulos disponibles.  

---
# Solución - Entrenamiento y Evaluación

- Se dividen los datos de **CUBDL** para el entrenamiento/validación en un 90%/10%. Se utilizan las adquisiciones de **PICMUS** para el dataset de prueba.

- Para comparar calidad de imagen, se utilizan las métricas de contraste CNR y gCNR.

- La red utiliza MSLE como función de pérdida para el entrenamiento y se utilizan 50 epocas.

---
# Resultados - Contrast speckle experimental

![center w:950 h:550](image-13.png)

---

![center w:900 h:300](image-14.png)

---
# Resultados - Contrast speckle simulado

![center w:950 h:550](image-15.png)

---

![center w:900 h:300](image-16.png)

---
# Resultados - Vista longitudinal de la carótida de PICMUS.

![center w:950 h:550](image-17.png)

---
# Resultados - Tabla MSLE dataset PICMUS

![center w:900 h:400](image-18.png)

---
# Discusión de los resultados

- **Calidad**: beamforming temprano (Modelo-1) logra la reconstrucción más coherente y con menor error.

- **MSLE**: BINN alcanza 0,00067 frente al 0,00081 del Modelo-1 en MSLE, pero, excluyendo los datos simulados por Field II, el promedio baja a 0,00035 con menos épocas y datos de entrenamiento.

- **Eficiencia/riesgos**: BINN es más rápido que todos los modelos y persiste riesgo de overfitting debido a la naturaleza de un grupo de datos de **CUBDL**.

---
# Conclusion

El Modelo-1, que integra la capa beamformer al inicio de la red, ofrece reconstrucciones más fieles al ground truth, obtiene los mejores MSLE y mejor contraste que los Modelos 2 y 3, confirmando que incorporar el beamforming en la arquitectura es prometedor, aunque los tiempos de inferencia aún no alcanzan los estandares de ultrafast imaging.

---
# Trabajo futuro

- **Optimizar DAS**: acelerar y aligerar la capa DAS, hoy cuello de botella en tiempo y memoria.

- **Ampliar dataset**: incorporar más adquisiciones, con distintos transductores y distintos objetivos para reducir overfitting.

- **Arquitecturas mejores**: evaluar redes más profundas/modernas (p. ej., ResNet/UNet-variantes) en lugar del modelo base simple.
