import streamlit as st
import cv2
import numpy as np
import easyocr
import sqlite3
from datetime import datetime
import torch
from ultralytics import YOLO
import time
import os
from pathlib import Path
import re
import threading

# Configuração inicial
st.set_page_config(page_title="Sistema de Monitoramento de Veículos", layout="wide")

# Criar pasta para salvar as imagens se não existir
SAVE_DIR = Path("fotos/detector de veiculos")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Inicializar variáveis de estado globais
if 'detection_active' not in st.session_state:
    st.session_state.detection_active = False
if 'notification' not in st.session_state:
    st.session_state.notification = None
if 'notification_time' not in st.session_state:
    st.session_state.notification_time = None
if 'stats_page' not in st.session_state:
    st.session_state.stats_page = 0
if 'take_photo' not in st.session_state:
    st.session_state.take_photo = False
if 'current_detections' not in st.session_state:
    st.session_state.current_detections = []
if 'current_session_plates' not in st.session_state:
    st.session_state.current_session_plates = {}
if 'total_placas' not in st.session_state:
    st.session_state.total_placas = 0
if 'total_placas_repetidas' not in st.session_state:
    st.session_state.total_placas_repetidas = 0
if 'latest_data' not in st.session_state:
    st.session_state.latest_data = []
if 'button_counter' not in st.session_state:
    st.session_state.button_counter = 0

# Variáveis globais para comunicação entre threads
global_stats = {
    'total_placas': 0,
    'total_placas_repetidas': 0,
    'latest_data': []
}
thread_running = False
stop_thread = False

# Função para gerar chaves únicas para botões
def get_unique_key(prefix):
    st.session_state.button_counter += 1
    return f"{prefix}_{st.session_state.button_counter}"

# Dicionário de classes
CLASSES = {
    0: ((0, 255, 0), 'pessoa'),    # Verde para pessoas
    2: ((255, 0, 0), 'carro'),     # Vermelho para carros
    3: ((0, 0, 255), 'moto'),      # Azul para motos
    5: ((255, 255, 0), 'caminhao'), # Amarelo para caminhões
    7: ((255, 0, 255), 'van'),     # Roxo para vans
    4: ((0, 255, 255), 'aviao'),   # Ciano para aviões
    6: ((128, 0, 128), 'onibus'),  # Roxo para ônibus
    1: ((255, 165, 0), 'bicicleta') # Laranja para bicicletas
}

def format_plate(plate):
    """
    Formata a placa adicionando hífen no formato correto
    """
    plate = plate.strip().upper()
    if len(plate) == 7:
        return f"{plate[:3]}-{plate[3:]}"
    return plate

def is_valid_brazilian_plate(plate):
    """
    Valida se uma string corresponde aos padrões de placa brasileira
    Suporta padrões Mercosul e antigo:
    - Padrão antigo: ABC1234 ou ABC-1234
    - Padrão Mercosul: ABC1D23 ou ABC-1D23
    """
    plate = plate.replace("-", "").strip().upper()
    old_pattern = r'^[A-Z]{3}[0-9]{4}$'
    mercosul_pattern = r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$'
    return bool(re.match(old_pattern, plate) or re.match(mercosul_pattern, plate))

def clean_plate_text(text):
    """
    Limpa e formata o texto da placa
    """
    text = re.sub(r'[^A-Z0-9-]', '', text.upper())
    return text

def init_db():
    conn = sqlite3.connect('veiculos.db', detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS veiculos (
            placa TEXT PRIMARY KEY,
            tipo TEXT,
            cor TEXT,
            modelo TEXT,
            contagem INTEGER,
            ultima_vista TIMESTAMP
        )
    ''')
    conn.commit()
    return conn

@st.cache_resource
def load_models():
    try:
        model = YOLO('yolov8n.pt')
        reader = easyocr.Reader(['pt'])
        return model, reader
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {str(e)}")
        return None, None

def detect_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    color_ranges = {
        'vermelho': ([0, 100, 100], [10, 255, 255]),
        'amarelo': ([20, 100, 100], [30, 255, 255]),
        'verde': ([40, 40, 40], [80, 255, 255]),
        'azul': ([90, 50, 50], [130, 255, 255]),
        'branco': ([0, 0, 200], [180, 30, 255]),
        'preto': ([0, 0, 0], [180, 255, 30]),
        'cinza': ([0, 0, 40], [180, 30, 200])
    }
    
    max_count = 0
    detected_color = "desconhecida"
    
    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        count = cv2.countNonZero(mask)
        
        if count > max_count:
            max_count = count
            detected_color = color
    
    return detected_color

def update_database_stats(conn):
    """
    Atualiza as estatísticas do banco de dados periodicamente
    """
    global global_stats, thread_running, stop_thread
    
    thread_running = True
    try:
        while not stop_thread:
            try:
                cursor = conn.cursor()
                
                # Atualizar estatísticas globais
                cursor.execute("SELECT COUNT(*) FROM veiculos")
                total_placas = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM veiculos WHERE contagem > 1")
                total_placas_repetidas = cursor.fetchone()[0]
                
                # Atualizar dados mais recentes
                cursor.execute("""
                    SELECT placa, tipo, cor, contagem, ultima_vista 
                    FROM veiculos 
                    ORDER BY ultima_vista DESC
                    LIMIT 5 OFFSET ?
                """, (0,))  # Sempre buscar a primeira página
                
                latest_data = cursor.fetchall()
                
                # Atualizar as variáveis globais de forma thread-safe
                global_stats = {
                    'total_placas': total_placas,
                    'total_placas_repetidas': total_placas_repetidas,
                    'latest_data': latest_data
                }
                
                # Pequena pausa para não sobrecarregar o banco de dados
                time.sleep(0.5)
            except Exception as e:
                print(f"Erro na consulta ao banco de dados: {str(e)}")
                time.sleep(1)
    except Exception as e:
        print(f"Erro na thread de atualização: {str(e)}")
    finally:
        thread_running = False

def process_frame(frame, model, reader, conn):
    if frame is None:
        return None, [], []
    
    try:
        results = model(frame, conf=0.5)  # Aumentar confiança mínima
        frame_draw = frame.copy()
        detected_plates = []
        current_detections = []
        
        if len(results) > 0 and len(results[0].boxes) > 0:
            for det in results[0].boxes.data:
                x1, y1, x2, y2, conf, cls = map(float, det)
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cls = int(cls)
                
                if cls in CLASSES:
                    cor, tipo = CLASSES[cls]
                    current_detections.append(tipo)
                    
                    if y2 > y1 and x2 > x1 and y1 >= 0 and x1 >= 0:
                        roi = frame[y1:y2, x1:x2]
                        
                        if roi.size > 0:
                            # Criar máscara para o objeto
                            mask = np.zeros_like(frame_draw)
                            contours = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
                            cv2.fillPoly(mask, [contours], cor)
                            
                            # Aplicar máscara com transparência
                            cv2.addWeighted(mask, 0.3, frame_draw, 1, 0, frame_draw)
                            
                            # Desenhar contorno
                            cv2.polylines(frame_draw, [contours], True, cor, 2)
                            
                            # Adicionar texto do tipo de objeto
                            cv2.putText(frame_draw, tipo, (x1, y1 - 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
                            
                            veiculo_cor = "N/A"
                            if tipo != 'pessoa':
                                veiculo_cor = detect_color(roi)
                                
                                try:
                                    # Melhorar detecção de placa
                                    placa_results = reader.readtext(roi, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-')
                                    placa_results = [r for r in placa_results if r[2] > 0.6]  # Aumentar confiança
                                    
                                    if placa_results:
                                        potential_plate = clean_plate_text(placa_results[0][1])
                                        
                                        if is_valid_brazilian_plate(potential_plate):
                                            placa = format_plate(potential_plate)
                                            
                                            cursor = conn.cursor()
                                            cursor.execute("SELECT contagem FROM veiculos WHERE placa = ?", (placa,))
                                            result = cursor.fetchone()
                                            
                                            # Definir cor da placa baseado no histórico
                                            if placa in st.session_state.current_session_plates:
                                                cor_placa = (255, 0, 0)  # Vermelho para placas já vistas na sessão
                                            elif result:
                                                cor_placa = (255, 0, 255)  # Roxo para placas já vistas antes
                                            else:
                                                cor_placa = (0, 255, 0)  # Verde para novas placas
                                            
                                            if result:
                                                nova_contagem = result[0] + 1
                                                cursor.execute("""
                                                    UPDATE veiculos 
                                                    SET contagem = ?, ultima_vista = ?, tipo = ?, cor = ?
                                                    WHERE placa = ?
                                                """, (nova_contagem, datetime.now(), tipo, veiculo_cor, placa))
                                            else:
                                                cursor.execute("""
                                                    INSERT INTO veiculos (placa, tipo, cor, contagem, ultima_vista)
                                                    VALUES (?, ?, ?, 1, ?)
                                                """, (placa, tipo, veiculo_cor, datetime.now()))
                                            
                                            conn.commit()
                                            detected_plates.append((placa, tipo, veiculo_cor))
                                            
                                            # Desenhar placa com fundo
                                            texto = f"{placa}"
                                            font = cv2.FONT_HERSHEY_SIMPLEX
                                            font_scale = 0.7
                                            thickness = 2
                                            
                                            # Obter tamanho do texto
                                            (text_width, text_height), _ = cv2.getTextSize(texto, font, font_scale, thickness)
                                            
                                            # Coordenadas para o fundo
                                            text_x = x1
                                            text_y = y1 - 10
                                            padding = 5
                                            
                                            # Desenhar fundo branco
                                            cv2.rectangle(frame_draw,
                                                        (text_x - padding, text_y - text_height - padding),
                                                        (text_x + text_width + padding, text_y + padding),
                                                        (255, 255, 255),
                                                        -1)
                                            
                                            # Desenhar borda colorida
                                            cv2.rectangle(frame_draw,
                                                        (text_x - padding, text_y - text_height - padding),
                                                        (text_x + text_width + padding, text_y + padding),
                                                        cor_placa,
                                                        2)
                                            
                                            # Desenhar texto
                                            cv2.putText(frame_draw, texto,
                                                      (text_x, text_y),
                                                      font, font_scale, (0, 0, 0), thickness)
                                except Exception as e:
                                    pass
        
        return frame_draw, detected_plates, current_detections
    
    except Exception as e:
        print(f"Erro ao processar frame: {str(e)}")
        return frame, [], []

def main():
    global stop_thread, thread_running
    
    st.title("Sistema de Monitoramento de Veículos")
    
    conn = init_db()
    model, reader = load_models()
    
    if model is None or reader is None:
        st.error("Não foi possível carregar os modelos necessários.")
        return
    
    # Layout principal com duas colunas
    col1, col2 = st.columns([3, 2])
    
    # Coluna da esquerda para vídeo e controles
    with col1:
        fonte = st.radio(
            "Selecione a fonte do vídeo:",
            ["Câmera", "Upload de Vídeo"],
            key="fonte_video"
        )
        
        # Botões de controle
        if not st.session_state.detection_active:
            if st.button("Iniciar Detecção", use_container_width=True):
                st.session_state.detection_active = True
                st.session_state.current_session_plates = {}
                stop_thread = False
                st.rerun()
        else:
            col_btns = st.columns(2)
            with col_btns[0]:
                if st.button("Parar Detecção", use_container_width=True):
                    st.session_state.detection_active = False
                    stop_thread = True
                    st.rerun()
            with col_btns[1]:
                if st.button("Tirar Foto", use_container_width=True):
                    st.session_state.take_photo = True
        
        # Notificações
        if st.session_state.notification:
            current_time = time.time()
            if st.session_state.notification_time and current_time - st.session_state.notification_time < 5:
                st.info(st.session_state.notification)
            else:
                st.session_state.notification = None
        
        # Placeholder para o vídeo
        video_placeholder = st.empty()
    
    # Coluna da direita para estatísticas
    with col2:
        # Criar placeholders para cada seção
        st.subheader("Detecções Atuais")
        detections_placeholder = st.empty()
        
        st.subheader("Contadores")
        counters_placeholder = st.empty()
        
        st.subheader("Placas Detectadas (Sessão Atual)")
        plates_placeholder = st.empty()
        
        st.subheader("Estatísticas")
        stats_nav_placeholder = st.empty()
        stats_data_placeholder = st.empty()
    
    # Iniciar detecção se ativo
    if st.session_state.detection_active:
        # Iniciar thread para atualizar estatísticas se ainda não estiver rodando
        if not thread_running:
            stats_thread = threading.Thread(target=update_database_stats, args=(conn,), daemon=True)
            stats_thread.start()
        
        if fonte == "Câmera":
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Não foi possível abrir a câmera.")
                st.session_state.detection_active = False
                stop_thread = True
                st.rerun()
            
            stframe = video_placeholder.empty()
            
            while st.session_state.detection_active:
                ret, frame = cap.read()
                if not ret:
                    st.error("Não foi possível ler o frame da câmera.")
                    break
                
                frame_processed, detected_plates, current_detections = process_frame(frame, model, reader, conn)
                
                # Atualizar detecções atuais
                st.session_state.current_detections = current_detections
                
                for placa, tipo, cor in detected_plates:
                    if placa in st.session_state.current_session_plates:
                        st.session_state.current_session_plates[placa] = (
                            st.session_state.current_session_plates[placa][0] + 1, tipo, cor
                        )
                    else:
                        st.session_state.current_session_plates[placa] = (1, tipo, cor)
                
                stframe.image(frame_processed, channels="BGR", caption="Detecção em tempo real", use_container_width=True)
                
                if st.session_state.take_photo:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"captura_manual_{timestamp}.jpg"
                    filepath = SAVE_DIR / filename
                    cv2.imwrite(str(filepath), frame)
                    st.session_state.notification = f"Imagem salva em {filepath}"
                    st.session_state.notification_time = time.time()
                    st.session_state.take_photo = False
                
                # Atualizar estatísticas em tempo real
                # 1. Detecções atuais
                with detections_placeholder.container():
                    if st.session_state.current_detections:
                        # Contar ocorrências de cada tipo
                        detection_counts = {}
                        for det in st.session_state.current_detections:
                            if det in detection_counts:
                                detection_counts[det] += 1
                            else:
                                detection_counts[det] = 1
                        
                        # Mostrar contagens
                        for tipo, count in detection_counts.items():
                            st.write(f"**{tipo.capitalize()}**: {count}")
                    else:
                        st.write("Nenhuma detecção no momento")
                
                # 2. Contadores - usar os dados da variável global
                with counters_placeholder.container():
                    col_counters = st.columns(2)
                    
                    with col_counters[0]:
                        st.metric("Total de Placas", global_stats['total_placas'])
                        st.metric("Placas Repetidas", global_stats['total_placas_repetidas'])
                    
                    with col_counters[1]:
                        st.metric("Placas na Sessão", len(st.session_state.current_session_plates))
                        st.metric("Repetidas na Sessão", sum(1 for _, (count, _, _) in st.session_state.current_session_plates.items() if count > 1))
                
                # 3. Placas detectadas
                with plates_placeholder.container():
                    if st.session_state.current_session_plates:
                        sorted_plates = sorted(st.session_state.current_session_plates.items(), 
                                              key=lambda x: x[1][0], reverse=True)
                        
                        for placa, (count, tipo, cor) in sorted_plates:
                            st.write(f"**{placa}** - {tipo} ({cor}) - {count}x")
                    else:
                        st.write("Nenhuma placa detectada na sessão atual.")
                
                # 4. Navegação de estatísticas - Usando chaves únicas
                with stats_nav_placeholder.container():
                    stats_cols = st.columns(3)
                    
                    with stats_cols[0]:
                        if st.button("⬆️ Topo", key=get_unique_key("btn_top_cam")):
                            st.session_state.stats_page = 0
                    
                    with stats_cols[1]:
                        if st.session_state.stats_page > 0:
                            if st.button("⬅️ Anterior", key=get_unique_key("btn_prev_cam")):
                                st.session_state.stats_page -= 1
                    
                    with stats_cols[2]:
                        if st.button("➡️ Próxima", key=get_unique_key("btn_next_cam")):
                            st.session_state.stats_page += 1
                
                # 5. Dados de estatísticas
                with stats_data_placeholder.container():
                    # Usar os dados da variável global
                    dados = global_stats['latest_data']
                    
                    if dados:
                        for d in dados:
                            st.write(f"**{d[0]}** - {d[1]} ({d[2]}) - Visto {d[3]}x - Última vez: {d[4]}")
                        st.caption(f"Página {st.session_state.stats_page + 1}")
                    else:
                        st.write("Nenhum veículo registrado ou fim dos registros.")
                        if st.session_state.stats_page > 0:
                            st.session_state.stats_page -= 1
                
                # Usar um pequeno delay para evitar travamentos
                time.sleep(0.01)
            
            cap.release()
            stop_thread = True
        
        else:
            video_file = st.file_uploader("Faça upload do vídeo", type=['mp4', 'avi'], key="video_upload")
            
            if video_file is not None:
                temp_file = f"temp_video_{int(time.time())}.mp4"
                with open(temp_file, "wb") as f:
                    f.write(video_file.getbuffer())
                
                cap = cv2.VideoCapture(temp_file)
                if not cap.isOpened():
                    st.error("Não foi possível abrir o vídeo.")
                    st.session_state.detection_active = False
                    stop_thread = True
                    st.rerun()
                
                stframe = video_placeholder.empty()
                
                while st.session_state.detection_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.info("Fim do vídeo.")
                        st.session_state.detection_active = False
                        break
                    
                    frame_processed, detected_plates, current_detections = process_frame(frame, model, reader, conn)
                    
                    # Atualizar detecções atuais
                    st.session_state.current_detections = current_detections
                    
                    for placa, tipo, cor in detected_plates:
                        if placa in st.session_state.current_session_plates:
                            st.session_state.current_session_plates[placa] = (
                                st.session_state.current_session_plates[placa][0] + 1, tipo, cor
                            )
                        else:
                            st.session_state.current_session_plates[placa] = (1, tipo, cor)
                    
                    stframe.image(frame_processed, channels="BGR", caption="Detecção em tempo real", use_container_width=True)
                    
                    # Atualizar estatísticas em tempo real (mesmo código que acima)
                    # 1. Detecções atuais
                    with detections_placeholder.container():
                        if st.session_state.current_detections:
                            # Contar ocorrências de cada tipo
                            detection_counts = {}
                            for det in st.session_state.current_detections:
                                if det in detection_counts:
                                    detection_counts[det] += 1
                                else:
                                    detection_counts[det] = 1
                            
                            # Mostrar contagens
                            for tipo, count in detection_counts.items():
                                st.write(f"**{tipo.capitalize()}**: {count}")
                        else:
                            st.write("Nenhuma detecção no momento")
                    
                    # 2. Contadores - usar os dados da variável global
                    with counters_placeholder.container():
                        col_counters = st.columns(2)
                        
                        with col_counters[0]:
                            st.metric("Total de Placas", global_stats['total_placas'])
                            st.metric("Placas Repetidas", global_stats['total_placas_repetidas'])
                        
                        with col_counters[1]:
                            st.metric("Placas na Sessão", len(st.session_state.current_session_plates))
                            st.metric("Repetidas na Sessão", sum(1 for _, (count, _, _) in st.session_state.current_session_plates.items() if count > 1))
                    
                    # 3. Placas detectadas
                    with plates_placeholder.container():
                        if st.session_state.current_session_plates:
                            sorted_plates = sorted(st.session_state.current_session_plates.items(), 
                                                key=lambda x: x[1][0], reverse=True)
                            
                            for placa, (count, tipo, cor) in sorted_plates:
                                st.write(f"**{placa}** - {tipo} ({cor}) - {count}x")
                        else:
                            st.write("Nenhuma placa detectada na sessão atual.")
                    
                    # 4. Navegação de estatísticas - Usando chaves únicas
                    with stats_nav_placeholder.container():
                        stats_cols = st.columns(3)
                        
                        with stats_cols[0]:
                            if st.button("⬆️ Topo", key=get_unique_key("btn_top_vid")):
                                st.session_state.stats_page = 0
                        
                        with stats_cols[1]:
                            if st.session_state.stats_page > 0:
                                if st.button("⬅️ Anterior", key=get_unique_key("btn_prev_vid")):
                                    st.session_state.stats_page -= 1
                        
                        with stats_cols[2]:
                            if st.button("➡️ Próxima", key=get_unique_key("btn_next_vid")):
                                st.session_state.stats_page += 1
                    
                    # 5. Dados de estatísticas
                    with stats_data_placeholder.container():
                        # Usar os dados da variável global
                        dados = global_stats['latest_data']
                        
                        if dados:
                            for d in dados:
                                st.write(f"**{d[0]}** - {d[1]} ({d[2]}) - Visto {d[3]}x - Última vez: {d[4]}")
                            st.caption(f"Página {st.session_state.stats_page + 1}")
                        else:
                            st.write("Nenhum veículo registrado ou fim dos registros.")
                            if st.session_state.stats_page > 0:
                                st.session_state.stats_page -= 1
                    
                    time.sleep(0.03)
                
                cap.release()
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                stop_thread = True
    else:
        # Mostrar estatísticas quando não estiver detectando
        with detections_placeholder.container():
            st.write("Detecção inativa")
        
        with counters_placeholder.container():
            col_counters = st.columns(2)
            
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM veiculos")
            total_placas = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM veiculos WHERE contagem > 1")
            total_placas_repetidas = cursor.fetchone()[0]
            
            with col_counters[0]:
                st.metric("Total de Placas", total_placas)
                st.metric("Placas Repetidas", total_placas_repetidas)
            
            with col_counters[1]:
                st.metric("Placas na Sessão", len(st.session_state.current_session_plates))
                st.metric("Repetidas na Sessão", sum(1 for _, (count, _, _) in st.session_state.current_session_plates.items() if count > 1))
        
        with plates_placeholder.container():
            if st.session_state.current_session_plates:
                sorted_plates = sorted(st.session_state.current_session_plates.items(), 
                                      key=lambda x: x[1][0], reverse=True)
                
                for placa, (count, tipo, cor) in sorted_plates:
                    st.write(f"**{placa}** - {tipo} ({cor}) - {count}x")
            else:
                st.write("Nenhuma placa detectada na sessão anterior.")
        
        with stats_nav_placeholder.container():
            stats_cols = st.columns(3)
            
            with stats_cols[0]:
                if st.button("⬆️ Topo", key=get_unique_key("btn_top_inactive")):
                    st.session_state.stats_page = 0
                    st.rerun()
            
            with stats_cols[1]:
                if st.session_state.stats_page > 0:
                    if st.button("⬅️ Anterior", key=get_unique_key("btn_prev_inactive")):
                        st.session_state.stats_page -= 1
                        st.rerun()
            
            with stats_cols[2]:
                if st.button("➡️ Próxima", key=get_unique_key("btn_next_inactive")):
                    st.session_state.stats_page += 1
                    st.rerun()
        
        with stats_data_placeholder.container():
            offset = st.session_state.stats_page * 5
            
            cursor = conn.cursor()
            cursor.execute("""
                SELECT placa, tipo, cor, contagem, ultima_vista 
                FROM veiculos 
                ORDER BY ultima_vista DESC
                LIMIT 5 OFFSET ?
            """, (offset,))
            
            dados = cursor.fetchall()
            
            if dados:
                for d in dados:
                    st.write(f"**{d[0]}** - {d[1]} ({d[2]}) - Visto {d[3]}x - Última vez: {d[4]}")
                st.caption(f"Página {st.session_state.stats_page + 1}")
            else:
                st.write("Nenhum veículo registrado ou fim dos registros.")
                if st.session_state.stats_page > 0:
                    st.session_state.stats_page -= 1
                    st.rerun()

if __name__ == "__main__":
    main()