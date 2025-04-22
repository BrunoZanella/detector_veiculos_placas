# Sistema de Monitoramento de Veículos

Este é um aplicativo Streamlit para detecção e monitoramento de veículos e placas em tempo real.

## Funcionalidades

- Detecção de veículos usando YOLOv8
- Reconhecimento de placas usando EasyOCR
- Monitoramento em tempo real via câmera ou vídeo
- Armazenamento de dados em banco SQLite
- Estatísticas de detecção

## Como usar

### Execução Local

1. Clone este repositório
2. Instale as dependências: `pip install -r requirements.txt`
3. Execute o aplicativo: `streamlit run app.py`

### No Streamlit Cloud

O aplicativo está disponível em: [veiculos.streamlit.app](https://veiculos.streamlit.app)

**Nota**: O acesso à câmera pode não funcionar no Streamlit Cloud. Recomendamos usar o modo de upload de vídeo quando estiver usando o aplicativo na nuvem.

## Requisitos

- Python 3.8+
- Streamlit
- OpenCV
- EasyOCR
- PyTorch
- Ultralytics (YOLOv8)

## Limitações no Streamlit Cloud

- O acesso à câmera pode não funcionar corretamente
- O armazenamento de dados é temporário e será perdido quando o aplicativo for reiniciado
- O processamento é mais lento devido à ausência de GPU
\`\`\`

## Principais Correções para o Streamlit Cloud

1. **Detecção de Ambiente Cloud**: Adicionei verificações para detectar quando o aplicativo está rodando no Streamlit Cloud.

2. **Armazenamento Temporário**: Configurei o banco de dados SQLite para usar um diretório temporário no Streamlit Cloud.

3. **Tratamento de Erros Aprimorado**: Adicionei mais tratamento de erros e mensagens informativas para ajudar os usuários.

4. **Otimização de Desempenho**: Reduzi a frequência de processamento de frames para melhorar o desempenho no ambiente cloud.

5. **Aviso sobre Câmera**: Adicionei um aviso específico sobre as limitações de acesso à câmera no Streamlit Cloud.

6. **Persistência de Conexão**: Melhorei o gerenciamento da conexão com o banco de dados para evitar problemas de concorrência.

Para publicar este código no Streamlit Cloud:

1. Crie um repositório no GitHub com estes arquivos
2. Renomeie `app_cloud_fixed.py` para `app.py`
3. Acesse [share.streamlit.io](https://share.streamlit.io/) e conecte seu repositório
4. Implante o aplicativo

Estas alterações devem resolver os problemas que você está enfrentando no Streamlit Cloud, especialmente o problema de "nada acontecer" quando você clica em "Iniciar Detecção".

<Actions>
  <Action name="Implementar exportação de dados" description="Adicionar funcionalidade para exportar dados de detecções para CSV ou Excel" />
  <Action name="Adicionar dashboard de estatísticas" description="Criar visualizações gráficas das detecções por tipo de veículo e horário" />
  <Action name="Otimizar detecção de placas" description="Melhorar o algoritmo de reconhecimento de placas para maior precisão" />
  <Action name="Adicionar modo offline" description="Implementar funcionalidade para operar sem conexão à internet" />
  <Action name="Corrigir avisos de timestamp" description="Adicionar conversor de timestamp personalizado para SQLite" />
</Actions>

\`\`\`

