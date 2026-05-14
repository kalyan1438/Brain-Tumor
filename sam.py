from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table,
    TableStyle, HRFlowable, PageBreak
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# Output path
OUTPUT = "outputs/IEEE_Brain_Tumor_Project_Reference.pdf"

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    rightMargin=15*mm,
    leftMargin=15*mm,
    topMargin=15*mm,
    bottomMargin=15*mm
)

styles = getSampleStyleSheet()
W, H = A4

# Custom styles
def S(name, **kw):
    base = kw.pop('base', 'Normal')
    return ParagraphStyle(name, parent=styles[base], **kw)

title_style = S('MyTitle', base='Title', fontSize=20, textColor=colors.HexColor('#1a1a2e'), spaceAfter=4)
h1 = S('H1', base='Heading1', fontSize=14, textColor=colors.HexColor('#16213e'), spaceBefore=12, spaceAfter=4)
h2 = S('H2', base='Heading2', fontSize=11, textColor=colors.HexColor('#0f3460'), spaceBefore=8, spaceAfter=3)
body = S('Body', fontSize=9, leading=14, spaceAfter=4)
code_style = S('Code', fontName='Courier', fontSize=8, leading=11, textColor=colors.HexColor('#1a1a1a'), backColor=colors.HexColor('#f5f5f5'), spaceAfter=4)
bullet = S('Bullet', fontSize=9, leading=14, leftIndent=12, spaceAfter=2)
green_box = S('Green', fontSize=9, leading=13, textColor=colors.HexColor('#155724'), backColor=colors.HexColor('#d4edda'), spaceAfter=4)
red_box = S('Red', fontSize=9, leading=13, textColor=colors.HexColor('#721c24'), backColor=colors.HexColor('#f8d7da'), spaceAfter=4)
blue_box = S('Blue', fontSize=9, leading=13, textColor=colors.HexColor('#004085'), backColor=colors.HexColor('#cce5ff'), spaceAfter=4)

def hr():
    return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cccccc'), spaceAfter=6)

def section_box(text, color='#e8f4f8'):
    return Paragraph(text, S('SBox', fontSize=9, leading=13, backColor=colors.HexColor(color), textColor=colors.black, spaceAfter=4, leftIndent=6, rightIndent=6))

story = []

# ─── COVER ───
story.append(Spacer(1, 20*mm))
story.append(Paragraph("🧠 IEEE Brain Tumor Project", title_style))
story.append(Paragraph("Complete Reference Document", S('Sub', base='Normal', fontSize=13, textColor=colors.HexColor('#0f3460'), alignment=TA_CENTER, spaceAfter=4)))
story.append(Paragraph("Super-Resolution Enhanced Brain Tumor Classification Using EfficientNetB3 on Degraded MRI Images", S('TitleSub', base='Normal', fontSize=10, textColor=colors.HexColor('#555'), alignment=TA_CENTER, spaceAfter=6)))
story.append(hr())
story.append(Spacer(1, 4*mm))

# Quick info table
info_data = [
    ["Conference", "National IEEE"],
    ["Dataset", "Kaggle Brain Tumor MRI (7023 images, 4 classes)"],
    ["SR Models", "SRCNN vs EDSR (Comparative Study)"],
    ["Classifier", "EfficientNetB3 (Transfer Learning)"],
    ["Localization", "YOLOv8"],
    ["Explainability", "Grad-CAM"],
    ["Frontend", "React + Vite + Three.js → Vercel"],
    ["Backend", "Express.js + MongoDB → Railway"],
    ["AI Server", "FastAPI → HuggingFace Spaces"],
    ["Submission", "August 2025"],
]
t = Table(info_data, colWidths=[45*mm, 120*mm])
t.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#16213e')),
    ('TEXTCOLOR', (0,0), (0,-1), colors.white),
    ('BACKGROUND', (1,0), (1,-1), colors.HexColor('#f8f9fa')),
    ('FONTSIZE', (0,0), (-1,-1), 9),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ('ROWBACKGROUNDS', (1,0), (1,-1), [colors.HexColor('#f8f9fa'), colors.HexColor('#ffffff')]),
    ('LEFTPADDING', (0,0), (-1,-1), 6),
    ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
]))
story.append(t)
story.append(PageBreak())

# ─── SECTION 1: TEAM ───
story.append(Paragraph("1. Team Division", h1))
story.append(hr())

team_data = [
    ["Member", "Role", "Responsibilities"],
    ["Member 1\n(Team Lead)", "AI Core", "SRCNN training, EfficientNetB3, Degradation experiments, PSNR/SSIM, Training metrics"],
    ["Member 2\n(You)", "SR Pipeline +\nDetection", "EDSR implementation, YOLOv8, Grad-CAM, FastAPI integration, 3D coord mapping"],
    ["Member 3", "Frontend", "React + Vite UI, Three.js 3D Brain, API connections, Vercel deploy"],
    ["Member 4", "Auth + Paper", "Express.js, MongoDB, JWT auth, Literature review, IEEE formatting"],
]
t2 = Table(team_data, colWidths=[30*mm, 35*mm, 100*mm])
t2.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f3460')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTSIZE', (0,0), (-1,-1), 8),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('LEFTPADDING', (0,0), (-1,-1), 5),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
]))
story.append(t2)
story.append(Spacer(1, 4*mm))

# ─── SECTION 2: PIPELINE ───
story.append(Paragraph("2. Complete Pipeline", h1))
story.append(hr())

pipeline = [
    ["Step", "Module", "Input → Output", "Owner"],
    ["1", "Degradation", "Original MRI → LR_Testing", "M2 (You) ✅ Done"],
    ["2", "SRCNN", "LR_Testing → SRCNN_Testing", "M2 (You) ✅ Done"],
    ["3", "EDSR", "LR_Testing → EDSR_Testing", "M2 (You) ⬜ Next"],
    ["4", "EfficientNetB3", "All datasets → Accuracy tables", "M1 ⬜ Pending"],
    ["5", "YOLOv8", "Enhanced MRI → Bounding boxes", "M2 (You) ⬜ Pending"],
    ["6", "Grad-CAM", "Model + Image → Heatmap", "M1 ⬜ Pending"],
    ["7", "FastAPI", "All models → API endpoint", "M2 (You) ⬜ Pending"],
    ["8", "React + Three.js", "API results → 3D visualization", "M3 ⬜ Pending"],
    ["9", "Express + MongoDB", "Doctor login + trial count", "M4 ⬜ Pending"],
    ["10", "Deploy", "Vercel + Railway + HuggingFace", "All ⬜ Pending"],
]
t3 = Table(pipeline, colWidths=[10*mm, 30*mm, 80*mm, 40*mm])
t3.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f3460')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTSIZE', (0,0), (-1,-1), 8),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
    ('LEFTPADDING', (0,0), (-1,-1), 5),
    ('TOPPADDING', (0,0), (-1,-1), 3),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
]))
story.append(t3)
story.append(PageBreak())

# ─── SECTION 3: CURRENT RESULTS ───
story.append(Paragraph("3. Current Results (SRCNN)", h1))
story.append(hr())

story.append(Paragraph("SRCNN vs LR_Testing Metrics:", h2))
results_data = [
    ["Method", "PSNR (dB)", "SSIM", "Status"],
    ["LR (Degraded)", "Baseline", "Baseline", "Reference"],
    ["SRCNN Enhanced", "29.82", "0.918", "✅ Complete"],
    ["EDSR Enhanced", "Pending", "Pending", "⬜ Next Step"],
]
t4 = Table(results_data, colWidths=[50*mm, 40*mm, 40*mm, 35*mm])
t4.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#155724')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTSIZE', (0,0), (-1,-1), 9),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ('BACKGROUND', (0,2), (-1,2), colors.HexColor('#d4edda')),
    ('LEFTPADDING', (0,0), (-1,-1), 6),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ('ALIGN', (1,0), (-1,-1), 'CENTER'),
]))
story.append(t4)
story.append(Spacer(1, 4*mm))

story.append(Paragraph("Result Analysis:", h2))
story.append(Paragraph("• PSNR 29.82 dB = Good range (25-35 dB acceptable for medical imaging)", bullet))
story.append(Paragraph("• SSIM 0.918 = Excellent (>0.9 = excellent structural similarity)", bullet))
story.append(Paragraph("• EDSR expected to give PSNR ~31-33 dB and SSIM ~0.92-0.94", bullet))
story.append(Paragraph("• PSNR Paradox possible: EDSR higher PSNR but SRCNN higher classification accuracy — this is a valid paper finding!", bullet))

# ─── SECTION 4: PENDING TASKS ───
story.append(Paragraph("4. Pending Implementation Tasks", h1))
story.append(hr())

story.append(Paragraph("Your Tasks (Member 2):", h2))
tasks_you = [
    ["Priority", "Task", "Est. Time"],
    ["1 - Immediate", "EDSR model train + EDSR_Testing generate", "1-2 days"],
    ["2 - After EDSR", "PSNR/SSIM comparison (SRCNN vs EDSR)", "1 hour"],
    ["3 - After M1 trains", "YOLOv8 annotation + training", "1 week"],
    ["4 - After YOLOv8", "Grad-CAM implementation + IoU", "2-3 days"],
    ["5 - Integration", "FastAPI — all models in one endpoint", "3-4 days"],
    ["6 - Integration", "3D coordinate mapping YOLO → Three.js", "1 day"],
]
t5 = Table(tasks_you, colWidths=[40*mm, 90*mm, 30*mm])
t5.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#856404')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTSIZE', (0,0), (-1,-1), 8),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#fff3cd'), colors.white]),
    ('LEFTPADDING', (0,0), (-1,-1), 5),
    ('TOPPADDING', (0,0), (-1,-1), 3),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
]))
story.append(t5)
story.append(Spacer(1, 3*mm))

story.append(Paragraph("Member 1 Tasks:", h2))
tasks_m1 = [
    ["Priority", "Task", "Est. Time"],
    ["1", "EfficientNetB3 train on LR_Testing", "2-3 hours (Colab)"],
    ["2", "EfficientNetB3 train on SRCNN_Testing", "2-3 hours (Colab)"],
    ["3", "EfficientNetB3 train on EDSR_Testing (after you generate)", "2-3 hours"],
    ["4", "Classification report + confusion matrix all datasets", "1 day"],
    ["5", "Grad-CAM figures — with/without SR comparison", "2 days"],
]
t6 = Table(tasks_m1, colWidths=[15*mm, 115*mm, 35*mm])
t6.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f3460')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTSIZE', (0,0), (-1,-1), 8),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#e8f4f8'), colors.white]),
    ('LEFTPADDING', (0,0), (-1,-1), 5),
    ('TOPPADDING', (0,0), (-1,-1), 3),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
]))
story.append(t6)
story.append(PageBreak())

# ─── SECTION 5: IEEE PAPER STRUCTURE ───
story.append(Paragraph("5. IEEE Paper — What to Include", h1))
story.append(hr())

story.append(Paragraph("Tables Required:", h2))
paper_tables = [
    ["Table", "Content", "Data Source"],
    ["Table 1", "SR Quality: LR vs SRCNN vs EDSR (PSNR/SSIM)", "Your SR experiments"],
    ["Table 2", "Classification: LR vs SRCNN vs EDSR accuracy per class", "Member 1 experiments"],
    ["Table 3", "Existing methods comparison (VGG16, ResNet, DenseNet, EfficientNetB0, Ours)", "Literature + your results"],
    ["Table 4", "Degradation levels study (Mild/Medium/Severe)", "Ablation experiments"],
]
t7 = Table(paper_tables, colWidths=[20*mm, 85*mm, 55*mm])
t7.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f3460')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTSIZE', (0,0), (-1,-1), 8),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
    ('LEFTPADDING', (0,0), (-1,-1), 5),
    ('TOPPADDING', (0,0), (-1,-1), 3),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
]))
story.append(t7)
story.append(Spacer(1, 3*mm))

story.append(Paragraph("Figures Required:", h2))
figs = [
    ["Figure", "Content"],
    ["Figure 1", "System pipeline diagram (LR → SR → EfficientNetB3 → YOLOv8 → Grad-CAM → 3D)"],
    ["Figure 2", "Visual comparison: Original | LR Degraded | SRCNN | EDSR (side by side)"],
    ["Figure 3", "Grad-CAM: Without SR (scattered) vs With SR (focused on tumor)"],
    ["Figure 4", "Training loss curves for SRCNN and EDSR"],
    ["Figure 5", "Confusion matrix for EfficientNetB3 on best SR method"],
]
t8 = Table(figs, colWidths=[20*mm, 140*mm])
t8.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f3460')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTSIZE', (0,0), (-1,-1), 8),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
    ('LEFTPADDING', (0,0), (-1,-1), 5),
    ('TOPPADDING', (0,0), (-1,-1), 3),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
]))
story.append(t8)
story.append(Spacer(1, 3*mm))

story.append(Paragraph("Paper Sections:", h2))
sections = [
    ["Section", "Content", "Status"],
    ["Abstract", "Problem + Method + Dataset + Results numbers", "✅ Draft ready"],
    ["Introduction", "Brain tumor importance + DL gap + Your contributions", "⬜ August"],
    ["Related Work", "SRCNN, EDSR, EfficientNet, YOLOv8 papers cite", "⬜ August"],
    ["Methodology", "Degradation + SRCNN + EDSR + B3 + YOLOv8 + Grad-CAM", "⬜ August"],
    ["Experiments", "Dataset + Training settings + All tables", "⬜ August"],
    ["Results", "All figures + Analysis + Discussion", "⬜ August"],
    ["Conclusion", "Summary + Limitations + Future work", "⬜ August"],
]
t9 = Table(sections, colWidths=[35*mm, 105*mm, 25*mm])
t9.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f3460')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTSIZE', (0,0), (-1,-1), 8),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
    ('LEFTPADDING', (0,0), (-1,-1), 5),
    ('TOPPADDING', (0,0), (-1,-1), 3),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ('BACKGROUND', (0,1), (-1,1), colors.HexColor('#d4edda')),
]))
story.append(t9)
story.append(PageBreak())

# ─── SECTION 6: TECH STACK ───
story.append(Paragraph("6. Final Tech Stack", h1))
story.append(hr())

tech_data = [
    ["Layer", "Technology", "Deploy", "Purpose"],
    ["AI Models", "PyTorch", "-", "Framework"],
    ["SR", "SRCNN + EDSR", "HuggingFace", "MRI enhancement"],
    ["Classifier", "EfficientNetB3", "HuggingFace", "4-class tumor classification"],
    ["Detector", "YOLOv8", "HuggingFace", "Tumor localization"],
    ["Explainability", "Grad-CAM", "HuggingFace", "Why model decided"],
    ["AI Server", "FastAPI", "HuggingFace", "Model serving API"],
    ["Auth Gateway", "Express.js", "Railway", "Doctor login + trials"],
    ["Database", "MongoDB Atlas", "Cloud", "Doctor accounts"],
    ["Frontend", "React + Vite", "Vercel", "Doctor UI"],
    ["3D Brain", "Three.js + R3F", "Vercel", "3D tumor visualization"],
    ["PDF Report", "FPDF2", "HuggingFace", "Clinical report"],
]
t10 = Table(tech_data, colWidths=[35*mm, 40*mm, 35*mm, 55*mm])
t10.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f3460')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTSIZE', (0,0), (-1,-1), 8),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
    ('LEFTPADDING', (0,0), (-1,-1), 5),
    ('TOPPADDING', (0,0), (-1,-1), 3),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
]))
story.append(t10)
story.append(Spacer(1, 4*mm))

# Architecture flow
story.append(Paragraph("Communication Flow:", h2))
story.append(Paragraph("Doctor Browser (Vercel) → POST MRI → Express.js (Railway) → Auth check + Trial count → FastAPI (HuggingFace) → SRCNN/EDSR + EfficientNetB3 + YOLOv8 + Grad-CAM → JSON response → Express → React displays results + Three.js 3D brain", body))

# ─── SECTION 7: RISKS ───
story.append(Paragraph("7. Key Technical Risks", h1))
story.append(hr())

risks = [
    ["Risk", "Probability", "Solution"],
    ["EfficientNetB3 overfit (3 class small data)", "High", "Augmentation + Dropout 0.5 + Label smoothing"],
    ["EDSR hallucinations (fake tumor features)", "Medium", "Use MSE loss only, NOT perceptual loss"],
    ["PSNR Paradox (EDSR higher PSNR, lower accuracy)", "Low", "Report as novel finding in paper!"],
    ["Class imbalance (No Tumor = fewer images)", "High", "Class weights in CrossEntropyLoss"],
    ["Grad-CAM instability (different maps each run)", "Low", "Use SmoothGradCAMpp instead"],
    ["Colab disconnect during training", "High", "Save checkpoint every 10 epochs to Drive"],
    ["YOLOv8 annotation time (1311 images)", "High", "LabelImg tool, divide among team members"],
    ["Team members not contributing", "High", "Weekly deadlines, simple tasks, starter code"],
],
t11 = Table(risks[0], colWidths=[65*mm, 25*mm, 75*mm])
t11.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#721c24')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTSIZE', (0,0), (-1,-1), 7.5),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8d7da'), colors.white]),
    ('LEFTPADDING', (0,0), (-1,-1), 5),
    ('TOPPADDING', (0,0), (-1,-1), 3),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
]))
story.append(t11)
story.append(PageBreak())

# ─── SECTION 8: TIMELINE ───
story.append(Paragraph("8. Timeline", h1))
story.append(hr())

timeline = [
    ["Month", "You (M2)", "Member 1", "Members 3+4"],
    ["March\n(Now)", "✅ LR_Testing done\n✅ SRCNN done\n⬜ EDSR next", "⬜ EfficientNetB3\ntrain on all datasets", "⬜ React setup\n⬜ Express setup"],
    ["April", "⬜ YOLOv8 training\n⬜ Grad-CAM\n⬜ FastAPI setup", "⬜ Accuracy tables\n⬜ Grad-CAM figures\n⬜ Confusion matrix", "⬜ React UI pages\n⬜ MongoDB setup"],
    ["May", "⬜ FastAPI complete\n⬜ All models integrated\n⬜ API testing", "⬜ Ablation study\n⬜ Results analysis", "⬜ Three.js 3D brain\n⬜ JWT auth"],
    ["June", "⬜ Integration\n⬜ End-to-end testing\n⬜ Bug fixes", "⬜ Final experiments\n⬜ Figures finalize", "⬜ Deploy\n⬜ Integration test"],
    ["July", "⬜ Paper writing\n(Methodology)", "⬜ Paper writing\n(Results + Experiments)", "⬜ Paper writing\n(Lit review, Format)"],
    ["August", "✅ IEEE Submit!", "✅ IEEE Submit!", "✅ IEEE Submit!"],
]
t12 = Table(timeline, colWidths=[20*mm, 50*mm, 50*mm, 45*mm])
t12.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f3460')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTSIZE', (0,0), (-1,-1), 7.5),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
    ('LEFTPADDING', (0,0), (-1,-1), 4),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('BACKGROUND', (0,7), (-1,7), colors.HexColor('#d4edda')),
    ('FONTNAME', (0,7), (-1,7), 'Helvetica-Bold'),
]))
story.append(t12)
story.append(Spacer(1, 4*mm))

# ─── SECTION 9: PREREQUISITES ───
story.append(Paragraph("9. Prerequisites — What to Learn", h1))
story.append(hr())

prereq = [
    ["Member", "Topics", "Resources"],
    ["M1 + M2\n(DL Team)", "Python, PyTorch, OpenCV, SRCNN, EDSR, EfficientNetB3, FastAPI, Google Colab", "pytorch.org/tutorials, fast.ai, YouTube freecodecamp"],
    ["M3\n(Frontend)", "React + Vite, Three.js, @react-three/fiber, @react-three/drei, Tailwind CSS", "nextjs.org/learn, docs.pmnd.rs, YouTube Traversy"],
    ["M4\n(Auth)", "Node.js, Express.js, JWT, bcrypt, MongoDB, Mongoose", "YouTube Net Ninja, mongoosejs.com"],
    ["All\nMembers", "Git + GitHub, VS Code, Branch workflow, Pull requests", "youtu.be/RGOj5yH7evk"],
]
t13 = Table(prereq, colWidths=[20*mm, 75*mm, 70*mm])
t13.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f3460')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTSIZE', (0,0), (-1,-1), 8),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#f8f9fa'), colors.white]),
    ('LEFTPADDING', (0,0), (-1,-1), 5),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
]))
story.append(t13)
story.append(PageBreak())

# ─── SECTION 10: LOCKED DECISIONS ───
story.append(Paragraph("10. All Locked Decisions", h1))
story.append(hr())

locked = [
    ["Decision", "Value"],
    ["Paper Title", "Super-Resolution-Enhanced Brain Tumor Classification Using EfficientNetB3 on Degraded MRI Images"],
    ["SR Models", "SRCNN vs EDSR (comparative study inside paper)"],
    ["Classifier", "EfficientNetB3 (300x300 input, 12M params, nobody in reference papers used B3)"],
    ["Dataset", "Kaggle Brain Tumor MRI — Nickparvar (7023 images, 4 classes)"],
    ["Classes", "Glioma, Meningioma, Pituitary, No Tumor"],
    ["Framework", "PyTorch (not TensorFlow)"],
    ["Degradation", "Downsampling + Gaussian blur (simulate real clinical conditions)"],
    ["Metrics SR", "PSNR + SSIM"],
    ["Metrics Classify", "Accuracy + Precision + Recall + F1 + Confusion Matrix"],
    ["Metrics Localize", "YOLOv8 mAP50 + Grad-CAM IoU"],
    ["AI Deploy", "FastAPI on HuggingFace Spaces"],
    ["Auth Deploy", "Express.js + MongoDB on Railway"],
    ["Frontend Deploy", "React + Vite + Three.js on Vercel"],
    ["Auth Features", "Doctor login, JWT tokens, 5 free trials limit"],
    ["Conference", "National IEEE (August 2025 submit)"],
]
t14 = Table(locked, colWidths=[45*mm, 120*mm])
t14.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#155724')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTSIZE', (0,0), (-1,-1), 8),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor('#d4edda'), colors.white]),
    ('LEFTPADDING', (0,0), (-1,-1), 5),
    ('TOPPADDING', (0,0), (-1,-1), 3),
    ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
]))
story.append(t14)
story.append(Spacer(1, 4*mm))

# ─── SECTION 11: 3D MAPPING ───
story.append(Paragraph("11. 3D Coordinate Mapping Formula", h1))
story.append(hr())
story.append(Paragraph("YOLO pixel coordinates → Three.js 3D brain space:", h2))

coord_data = [
    ["Variable", "Formula", "Description"],
    ["brain_x", "(cx / img_size - 0.5) × 160", "Left-Right position"],
    ["brain_y", "(cy / img_size - 0.5) × 130", "Front-Back position"],
    ["brain_z", "0 (approximate)", "Depth (center slice)"],
    ["tumor_size", "((x2-x1)+(y2-y1)) / 2 / img_size × 40", "Sphere radius in Three.js"],
]
t15 = Table(coord_data, colWidths=[30*mm, 75*mm, 60*mm])
t15.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f3460')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
    ('FONTSIZE', (0,0), (-1,-1), 8),
    ('GRID', (0,0), (-1,-1), 0.3, colors.HexColor('#dee2e6')),
    ('FONTNAME', (1,1), (1,-1), 'Courier'),
    ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f8f9fa')),
    ('LEFTPADDING', (0,0), (-1,-1), 5),
    ('TOPPADDING', (0,0), (-1,-1), 4),
    ('BOTTOMPADDING', (0,0), (-1,-1), 4),
]))
story.append(t15)
story.append(Spacer(1, 4*mm))

# ─── FOOTER NOTE ───
story.append(hr())
story.append(Paragraph("📝 Note: Paper writing starts August. Focus on implementation first. EDSR is next immediate step.", 
    S('Note', base='Normal', fontSize=9, textColor=colors.HexColor('#856404'), backColor=colors.HexColor('#fff3cd'), 
      spaceAfter=4, leftIndent=6)))
story.append(Paragraph("🇩🇪 Germany MS Goal: IEEE paper + strong implementation = powerful SOP + CV for MS applications.", 
    S('Note2', base='Normal', fontSize=9, textColor=colors.HexColor('#004085'), backColor=colors.HexColor('#cce5ff'), 
      spaceAfter=4, leftIndent=6)))

doc.build(story)
print("PDF generated successfully!")