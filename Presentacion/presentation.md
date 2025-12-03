---
marp: true
title: "Neuroback"
theme: gaia
paginate: true
---

## NeuroBack: Análisis y lineas de investigación.
##### Integrantes:
- Matías Sandoval
- Nicolas Barahona
- Fernando Salgado
- Sebastián Gutierrez
- Sophia Escobar 
- José Pinto


---
### Punteo:
- Descripcion Neuroback
- Limitaciones actuales
- posibles lineas de investigaciones futuras
- Modelo (explicar modelo actual y cambios implementados)
- Heuristicas backbone (explicar por que son interesantes y detallarlas)
- Resultados
- discusión 
- conclusiones
--- 
## NeuroBack

- Objetivo: mejorar solvers CDCL para SAT usando Graph Neural Networks de forma práctica y eficiente.
- Busca predecir las "fases" mayoritarias de variables (backbone) con una GNN una sola vez antes de resolver y aplicar esas predicciones durante la búsqueda, evitando inferencias frecuentes en GPU mientras ejecuta el solver.

---

### NeuroBack: Estado del arte

Los métodos previos similares a Neuroback como Janota(2010), NeuroSat o NeuroCore:
- No mejoraban los resultados.
- Requerían frecuentes inferencias online en GPU.
Lo que inducia en un peor desempeño.
---

### NeuroBack: Método

- Representación: grafo CNF (nodos de variables y cláusulas, aristas según incidencia). 
![width:200px](Graph_cnf.png)
- GNN que predice la fase mayoritaria (Backbone) para cada variable.
- Las predicciones se fijan/inyectan como sugerencias iniciales para el solver; el solver sigue siendo CDCL pero con mejor punto de partida.

---

### NeuroBack 

- NeuroBack: enfoque que realiza una única inferencia offline con un GNN antes del run del solver.
- DataBack: nuevo dataset con 120,286 muestras para entrenar el modelo.
- Integración con el solver Kissat, mostrando mejoras en problemas resueltos (+5.2% en SATCOMP-2022, +7.4% en SATCOMP-2023).
- Demuestra que una inferencia offline bien entrenada puede ser suficiente para proporcionar señales útiles a heurísticas clásicas.

---

### Limitaciones actuales

---

### Modelo:

---

### Heurísticas

- Neuroback-Kissat funciona con Sistema cascada.
- Implementacion sencilla pero efectiva.
- Preexistentes
    - Neural-backbone  --initial (principal).
    - Neural-backbone  --always.
- Propuestas
    - Partial-backbone
    - Prioritized-backbone
    - LowScores-backbones

---

### Heurísticas: Initial/Always backbone

- Initial backbone
    - Utilización de la red solo para el signo inicial
- Heurística estocástica.
    - Priorización del uso de la red para todas las decisiones
- Admiten como posible variable de backbone si la confianza es alta
- Por cada una, se elige variable, se asigna fase, y se propaga

---

### Heurísticas: Partial backbone

- Heurística estocástica.
- Utilizar una fracción del backbone predicho.
- Puede ayudar en problemas donde el backbone predicho no es completamente correcto.

---

### Heurísticas: Prioritized backbone 

- Heurística de variable branching
- La cola de decisión se ordena según probabilidad de predicción
- Posterior a esto, se aplica phase selection

---

### Heurísticas: LowScores backbone 

- Kissat utiliza VSIDS como heurística. 
- Utilizar el backbone solo a variables poco activas.
    - Variables de alta actividad: VSIDS ha demostrado ser confiable → se sigue usando VSIDS.
    - Variables de baja actividad: VSIDS es menos informativo → aquí las predicciones de la GNN pueden aportar valor.

---


### Referencias
- Wenxi Wang et al., "NeuroBack: Improving CDCL SAT Solving using Graph Neural Networks", arXiv:2110.14053 
