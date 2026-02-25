import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

# ===== ตั้งค่าตรงนี้ =====
VEHICLE_MODEL_PATH = "vehicle_detector.pt"   # โมเดลแยกประเภทรถ
LICENSE_MODEL_PATH = "license_reader.pt"     # โมเดลอ่านป้ายทะเบียน
VEHICLE_CONF = 0.5
LICENSE_CONF = 0.5
CAMERA_INDEX = 0
# =========================

# -------- Thai province mapping --------
PROVINCE_MAP = {
    "กรุงเทพ": "กรุงเทพมหานคร",
    "กทม": "กรุงเทพมหานคร",
    "กระบี่": "กระบี่",
    "กาญจนบุรี": "กาญจนบุรี",
    "กาฬสินธุ์": "กาฬสินธุ์",
    "กำแพงเพชร": "กำแพงเพชร",
    "ขอนแก่น": "ขอนแก่น",
    "จันทบุรี": "จันทบุรี",
    "ฉะเชิงเทรา": "ฉะเชิงเทรา",
    "ชลบุรี": "ชลบุรี",
    "ชัยนาท": "ชัยนาท",
    "ชัยภูมิ": "ชัยภูมิ",
    "ชุมพร": "ชุมพร",
    "เชียงราย": "เชียงราย",
    "เชียงใหม่": "เชียงใหม่",
    "ตรัง": "ตรัง",
    "ตราด": "ตราด",
    "ตาก": "ตาก",
    "นครนายก": "นครนายก",
    "นครปฐม": "นครปฐม",
    "นครพนม": "นครพนม",
    "นครราชสีมา": "นครราชสีมา",
    "นครศรีธรรมราช": "นครศรีธรรมราช",
    "นครสวรรค์": "นครสวรรค์",
    "นนทบุรี": "นนทบุรี",
    "นราธิวาส": "นราธิวาส",
    "น่าน": "น่าน",
    "บึงกาฬ": "บึงกาฬ",
    "บุรีรัมย์": "บุรีรัมย์",
    "ปทุมธานี": "ปทุมธานี",
    "ประจวบคีรีขันธ์": "ประจวบคีรีขันธ์",
    "ปราจีนบุรี": "ปราจีนบุรี",
    "ปัตตานี": "ปัตตานี",
    "พระนครศรีอยุธยา": "พระนครศรีอยุธยา",
    "พะเยา": "พะเยา",
    "พังงา": "พังงา",
    "พัทลุง": "พัทลุง",
    "พิจิตร": "พิจิตร",
    "พิษณุโลก": "พิษณุโลก",
    "เพชรบุรี": "เพชรบุรี",
    "เพชรบูรณ์": "เพชรบูรณ์",
    "แพร่": "แพร่",
    "ภูเก็ต": "ภูเก็ต",
    "มหาสารคาม": "มหาสารคาม",
    "มุกดาหาร": "มุกดาหาร",
    "แม่ฮ่องสอน": "แม่ฮ่องสอน",
    "ยโสธร": "ยโสธร",
    "ยะลา": "ยะลา",
    "ร้อยเอ็ด": "ร้อยเอ็ด",
    "ระนอง": "ระนอง",
    "ระยอง": "ระยอง",
    "ราชบุรี": "ราชบุรี",
    "ลพบุรี": "ลพบุรี",
    "ลำปาง": "ลำปาง",
    "ลำพูน": "ลำพูน",
    "เลย": "เลย",
    "ศรีสะเกษ": "ศรีสะเกษ",
    "สกลนคร": "สกลนคร",
    "สงขลา": "สงขลา",
    "สตูล": "สตูล",
    "สมุทรปราการ": "สมุทรปราการ",
    "สมุทรสงคราม": "สมุทรสงคราม",
    "สมุทรสาคร": "สมุทรสาคร",
    "สระแก้ว": "สระแก้ว",
    "สระบุรี": "สระบุรี",
    "สิงห์บุรี": "สิงห์บุรี",
    "สุโขทัย": "สุโขทัย",
    "สุพรรณบุรี": "สุพรรณบุรี",
    "สุราษฎร์ธานี": "สุราษฎร์ธานี",
    "สุรินทร์": "สุรินทร์",
    "หนองคาย": "หนองคาย",
    "หนองบัวลำภู": "หนองบัวลำภู",
    "อ่างทอง": "อ่างทอง",
    "อำนาจเจริญ": "อำนาจเจริญ",
    "อุดรธานี": "อุดรธานี",
    "อุตรดิตถ์": "อุตรดิตถ์",
    "อุทัยธานี": "อุทัยธานี",
    "อุบลราชธานี": "อุบลราชธานี",
}

def extract_province(class_name: str) -> str:
    """ดึงชื่อจังหวัดจาก class name ของโมเดลป้ายทะเบียน"""
    for key, full_name in PROVINCE_MAP.items():
        if key in class_name:
            return full_name
    # ถ้าไม่เจอในลิสต์ ให้คืน class name ตรงๆ
    return class_name

def extract_plate_number(class_name: str) -> str:
    """ดึงเลขทะเบียนจาก class name (ปรับตามรูปแบบ class ของโมเดลคุณ)"""
    # ตัวอย่าง: ถ้า class name เป็น "กข-1234" หรือ "กข1234"
    # ปรับ logic ตรงนี้ให้ตรงกับ format ของโมเดลคุณ
    return class_name

@st.cache_resource
def load_models():
    vehicle_model = YOLO(VEHICLE_MODEL_PATH)
    license_model = YOLO(LICENSE_MODEL_PATH)
    return vehicle_model, license_model


def process_frame(frame, vehicle_model, license_model):
    """Process frame and return annotated frame + detection results"""
    results_data = []

    # --- Detect vehicles ---
    vehicle_results = vehicle_model(frame, conf=VEHICLE_CONF, verbose=False)[0]

    for vbox in vehicle_results.boxes:
        vx1, vy1, vx2, vy2 = map(int, vbox.xyxy[0])
        vehicle_class = vehicle_model.names[int(vbox.cls[0])]
        vehicle_conf = float(vbox.conf[0])

        # --- Crop รูปรถจาก vehicle_detector bbox ก่อนวาดกรอบ (ได้รูปสะอาดไม่มีกรอบ) ---
        pad = 6
        fh, fw = frame.shape[:2]
        cx1, cy1 = max(0, vx1 - pad), max(0, vy1 - pad)
        cx2, cy2 = min(fw, vx2 + pad), min(fh, vy2 + pad)
        vehicle_crop_img = frame[cy1:cy2, cx1:cx2].copy() if (cx2 > cx1 and cy2 > cy1) else None

        # วาดกรอบรถ (สีส้ม) — วาดหลัง crop แล้ว
        cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (255, 140, 0), 2)
        cv2.putText(frame, f"{vehicle_class} {vehicle_conf:.0%}",
                    (vx1, vy1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 140, 0), 2)

        # --- Detect license plate ภายในบริเวณรถ ---
        vehicle_crop = frame[vy1:vy2, vx1:vx2]
        plate_number = "ไม่พบป้าย"
        province = "-"

        if vehicle_crop.size > 0:
            license_results = license_model(vehicle_crop, conf=LICENSE_CONF, verbose=False)[0]

            for lbox in license_results.boxes:
                lx1, ly1, lx2, ly2 = map(int, lbox.xyxy[0])
                plate_class = license_model.names[int(lbox.cls[0])]

                # พิกัดป้ายบน frame หลัก
                abs_lx1, abs_ly1 = vx1 + lx1, vy1 + ly1
                abs_lx2, abs_ly2 = vx1 + lx2, vy1 + ly2

                # วาดกรอบป้ายทะเบียน (สีเขียว)
                cv2.rectangle(frame, (abs_lx1, abs_ly1), (abs_lx2, abs_ly2), (0, 220, 80), 2)
                cv2.putText(frame, plate_class,
                            (abs_lx1, abs_ly2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 80), 2)

                plate_number = extract_plate_number(plate_class)
                province = extract_province(plate_class)
                break

        results_data.append({
            "ประเภทรถ": vehicle_class,
            "ป้ายทะเบียน": plate_number,
            "จังหวัด": province,
            "ความมั่นใจ": f"{vehicle_conf:.0%}",
            "vehicle_crop_img": vehicle_crop_img,
        })

    return frame, results_data


# ===== Streamlit UI =====
st.set_page_config(
    page_title="Vehicle Detection",
    page_icon="🚗",
    layout="wide",
)

st.markdown("""
    <style>
        body { background-color: #0f1117; }
        .title { font-size: 2rem; font-weight: 700; color: #f0f0f0; }
        .detail-card {
            background: #1e2130;
            border-radius: 12px;
            padding: 16px 20px;
            margin-bottom: 10px;
            border-left: 4px solid #00d4aa;
        }
        .detail-label { color: #8899aa; font-size: 0.8rem; margin-bottom: 2px; }
        .detail-value { color: #ffffff; font-size: 1.1rem; font-weight: 600; }
        .badge {
            display: inline-block;
            background: #00d4aa22;
            color: #00d4aa;
            border: 1px solid #00d4aa55;
            border-radius: 6px;
            padding: 2px 10px;
            font-size: 0.85rem;
            font-weight: 600;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🚗 Vehicle & License Plate Detection</div>', unsafe_allow_html=True)
st.markdown("---")

# Layout
col_cam, col_info = st.columns([3, 2])

with col_cam:
    st.subheader("📷 กล้อง")
    frame_placeholder = st.empty()
    status_placeholder = st.empty()

with col_info:
    st.subheader("📋 รายละเอียดที่ตรวจพบ")
    info_placeholder = st.empty()
    # placeholder สำหรับแสดงรูปป้ายทะเบียนแต่ละคัน (สร้างแบบ dynamic ใน loop)

# Control buttons
col_start, col_stop = st.columns(2)
with col_start:
    start = st.button("▶️ เริ่มกล้อง", use_container_width=True, type="primary")
with col_stop:
    stop = st.button("⏹️ หยุด", use_container_width=True)

if "running" not in st.session_state:
    st.session_state.running = False

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

# Load models
if st.session_state.running:
    try:
        with st.spinner("กำลังโหลดโมเดล..."):
            vehicle_model, license_model = load_models()
        status_placeholder.success("โหลดโมเดลสำเร็จ ✅")
    except Exception as e:
        st.error(f"โหลดโมเดลไม่ได้: {e}")
        st.session_state.running = False

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        st.error("ไม่สามารถเปิดกล้องได้ กรุณาตรวจสอบ CAMERA_INDEX")
        st.session_state.running = False

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("ไม่สามารถอ่านภาพจากกล้องได้")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        annotated, detections = process_frame(frame_rgb, vehicle_model, license_model)

        # แสดงภาพ
        frame_placeholder.image(annotated, channels="RGB", use_container_width=True)

        # แสดงรายละเอียด
        with col_info:
            info_placeholder.empty()
            if detections:
                with info_placeholder.container():
                    for i, d in enumerate(detections):
                        st.markdown(f"""
                        <div class="detail-card">
                            <div style="margin-bottom:8px;">
                                <span class="badge">รถคันที่ {i+1}</span>
                                <span style="color:#8899aa; font-size:0.8rem; margin-left:8px;">ความมั่นใจ {d['ความมั่นใจ']}</span>
                            </div>
                            <div class="detail-label">ประเภทรถ</div>
                            <div class="detail-value">🚗 {d['ประเภทรถ']}</div>
                            <hr style="border-color:#ffffff11; margin:8px 0;">
                            <div class="detail-label">ป้ายทะเบียน</div>
                            <div class="detail-value">🔢 {d['ป้ายทะเบียน']}</div>
                            <hr style="border-color:#ffffff11; margin:8px 0;">
                            <div class="detail-label">จังหวัด</div>
                            <div class="detail-value">📍 {d['จังหวัด']}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # แสดงรูปรถที่ detect เจอจาก vehicle_detector
                        if d["vehicle_crop_img"] is not None:
                            st.image(
                                Image.fromarray(d["vehicle_crop_img"]),
                                caption=f"📸 รถคันที่ {i+1} — {d['ประเภทรถ']}",
                                use_container_width=True,
                            )
            else:
                info_placeholder.markdown("""
                    <div style="color:#8899aa; text-align:center; padding:40px 0;">
                        <div style="font-size:2rem;">🔍</div>
                        <div>ยังไม่พบรถในภาพ</div>
                    </div>
                """, unsafe_allow_html=True)

        time.sleep(0.03)

    cap.release()