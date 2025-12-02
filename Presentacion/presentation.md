---
marp: true
title: "Neuroback"
theme: gaia
paginate: true
---

## NeuroBack: Análisis y lineas de investigación.
##### Integrantes:
- Matías Sandoval
- Fernando Salgado
- Sebastián Gutierrez
- José Pinto
- Sophia Escobar 

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
### Referencias

- Wenxi Wang et al., "NeuroBack: Improving CDCL SAT Solving using Graph Neural Networks", arXiv:2110.14053 
