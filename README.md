# Federated-Learning

Understanding the Main Challenges of Federated Learning

## Final Report

The final report can be downloaded from here: [Report](report/final-report.pdf)

## Overview

Nowadays, a lot of data cannot be exploited by traditional Deep Learning (DL) methods due to its sensitive and privacy-protected nature. For instance, we can think of the data collected by the cameras or GPS sensors in our mobile phones, or produced by the Internet of Things (IoT). Introduced in 2016 by Google, Federated Learning (FL) is a machine learning scenario aiming to use privacy-protected data without violating the regulations in force. The goal is to learn a global model in privacy-constrained scenarios leveraging a client-server architecture. The central model is kept on the server-side and, unlike standard DL settings, has no direct access to the data, which is stored on multiple edge devices, i.e. the clients. Thanks to a paradigm based on the exchange of the model parameters (or gradient) between clients and server through multiple rounds, the global model is able to extract knowledge from data without breaking the users’ privacy.
The goal of this project is to become familiar with the standard federated scenario and understand its main challenges: i) statistical heterogeneity, i.e. the non-i.i.d. distribution of the clients’ data which leads to degraded performances and unstable learning; ii) systems heterogeneity, i.e. the presence of devices having different computational capabilities (e.g. smartphones VS servers); iii) privacy concerns, deriving from the possibility of obtaining information on clients’ data from the updated model exchanged on the network. Specifically for the latter, you will investigate the gradient inversion attack, i.e. recovering the input from the gradient of the trained model. Once these issues are understood, you will have the opportunity to propose your solution or to implement an existing one among those selected.

## Goals
1. To become familiar with the standard federated scenario and its main algorithms.
2. To understand the real-world challenges related to data statistical heterogeneity, systems heterogeneity and privacy concerns (specifically the gradient inversion attack).
3. To replicate the experiments detailed in the following sections.
4. To implement and test your contribution for solving one of the highlighted issues.
