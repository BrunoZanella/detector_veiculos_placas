# Sistema de Monitoramento de Veículos

Este é um sistema de monitoramento de veículos em tempo real que utiliza visão computacional para detectar e contar veículos, reconhecer placas e armazenar informações em um banco de dados SQLite.

## Funcionalidades

- Detecção de veículos em tempo real
- Reconhecimento de placas
- Contagem de veículos por tipo
- Armazenamento em banco de dados SQLite
- Interface web acessível via Streamlit
- Suporte para câmera ao vivo e upload de vídeos

## Requisitos

Instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

## Como executar

1. Instale as dependências
2. Execute o aplicativo:

```bash
streamlit run app.py
```

## Notas

- O sistema utiliza YOLO para detecção de veículos
- EasyOCR é usado para reconhecimento de placas
- Os dados são armazenados em um banco SQLite local
- A interface web é construída com Streamlit