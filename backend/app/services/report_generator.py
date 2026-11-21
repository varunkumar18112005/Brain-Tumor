from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, HRFlowable
)
from app.core.config import settings
from app.core.logger import logger

class ReportGeneratorService:
    @staticmethod
    def generate_pdf_report(
        analysis_data: Dict[str, Any],
        output_filename: Optional[str] = None
    ) -> str:
        """
        Builds a comprehensive, highly professional medical AI decision support report in PDF format.
        Includes scan details, 4 side-by-side processing images, findings, radiomic table,
        uncertainty breakdown, fuzzy logic rules fired, and medical disclaimers.
        """
        analysis_id = analysis_data.get("analysis_id", "N/A")
        patient_id = analysis_data.get("patient_id", "UNKNOWN")
        modality = analysis_data.get("modality", "MRI")
        created_at = analysis_data.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        pdf_filename = output_filename or f"Medical_Report_Analysis_{analysis_id}.pdf"
        pdf_path = settings.REPORTS_DIR / pdf_filename
        
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#0f3d64"),
            spaceAfter=4
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#555555"),
            spaceAfter=12
        )
        section_style = ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading2"],
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#0f3d64"),
            spaceBefore=8,
            spaceAfter=4
        )
        body_style = ParagraphStyle(
            "BodyDark",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#222222")
        )
        badge_style = ParagraphStyle(
            "Badge",
            parent=styles["Normal"],
            fontSize=10,
            leading=13,
            textColor=colors.white
        )
        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#777777")
        )

        elements = []

        # Header Title
        elements.append(Paragraph("INTELLIGENT MEDICAL IMAGE-BASED TUMOR DETECTION", title_style))
        elements.append(Paragraph("Decision Support & Uncertainty Assessment Clinical Research Report", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f3d64"), spaceAfter=10))

        # Metadata Table
        meta_data = [
            [
                Paragraph(f"<b>Patient Reference ID:</b> {patient_id}", body_style),
                Paragraph(f"<b>Analysis ID:</b> #{analysis_id}", body_style),
                Paragraph(f"<b>Imaging Modality:</b> {modality}", body_style)
            ],
            [
                Paragraph(f"<b>Scan Filename:</b> {analysis_data.get('scan_filename', 'scan.png')}", body_style),
                Paragraph(f"<b>Timestamp:</b> {created_at}", body_style),
                Paragraph(f"<b>Status:</b> Complete", body_style)
            ]
        ]
        meta_table = Table(meta_data, colWidths=[180, 180, 180])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f5f9")),
            ('PADDING', (0,0), (-1,-1), 6),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 10))

        # Section: Imaging Visualizations
        elements.append(Paragraph("1. Medical Image Acquisition & Pipeline Transformations", section_style))
        
        file_id = analysis_data.get("file_id") or str(analysis_id)
        subfolder = settings.UPLOADS_DIR / str(file_id)
        
        orig_img_path = subfolder / "original.png"
        enh_img_path = subfolder / "enhanced.png"
        mask_img_path = subfolder / "segmentation_mask.png"
        overlay_img_path = subfolder / "tumor_overlay.png"

        img_cells = []
        img_labels = []
        
        for p, label in [
            (orig_img_path, "Original Scan"),
            (enh_img_path, "Contrast Enhanced"),
            (mask_img_path, "Segmentation Mask"),
            (overlay_img_path, "Tumor Overlay")
        ]:
            if p.exists():
                img_cells.append(RLImage(str(p), width=120, height=120))
            else:
                img_cells.append(Paragraph(f"[{label} Pending]", body_style))
            img_labels.append(Paragraph(f"<b>{label}</b>", body_style))

        img_table = Table([img_cells, img_labels], colWidths=[135, 135, 135, 135])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,1), (-1,1), 4),
            ('BOTTOMPADDING', (0,0), (-1,0), 2),
        ]))
        elements.append(img_table)
        elements.append(Spacer(1, 10))

        # Section: Diagnostic Classification & Findings
        elements.append(Paragraph("2. Deep Learning Classification Findings", section_style))
        pred_class = analysis_data.get("predicted_class", "No Tumor")
        conf = float(analysis_data.get("confidence_score", 0.0))
        entropy = float(analysis_data.get("predictive_entropy", 0.0))
        tumor_detected = analysis_data.get("tumor_detected", False)
        
        class_color = "#dc2626" if tumor_detected else "#16a34a"
        
        findings_data = [
            [
                Paragraph("<b>Predicted Class:</b>", body_style),
                Paragraph(f"<font color='{class_color}'><b>{pred_class.upper()}</b></font>", body_style),
                Paragraph("<b>Tumor Detection:</b>", body_style),
                Paragraph(f"<b>{'POSITIVE' if tumor_detected else 'NEGATIVE'}</b>", body_style)
            ],
            [
                Paragraph("<b>Confidence Score:</b>", body_style),
                Paragraph(f"{conf*100:.1f}%", body_style),
                Paragraph("<b>Predictive Entropy:</b>", body_style),
                Paragraph(f"{entropy:.3f} (Shannon bits)", body_style)
            ]
        ]
        findings_table = Table(findings_data, colWidths=[120, 150, 120, 150])
        findings_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#94a3b8")),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
            ('PADDING', (0,0), (-1,-1), 5),
        ]))
        elements.append(findings_table)
        elements.append(Spacer(1, 8))

        # Class probabilities sub-table
        class_probs = analysis_data.get("class_probabilities", {})
        if class_probs:
            prob_rows = [["Tumor Class", "Probability", "Status"]]
            for k, v in class_probs.items():
                p_val = float(v)
                mark = "◄ Top Candidate" if k == pred_class else ""
                prob_rows.append([k, f"{p_val*100:.2f}%", mark])
            prob_table = Table(prob_rows, colWidths=[160, 160, 220])
            prob_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f3d64")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('PADDING', (0,0), (-1,-1), 4),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
            ]))
            elements.append(prob_table)
        elements.append(Spacer(1, 10))

        # Section: Dual Uncertainty Assessment
        elements.append(Paragraph("3. Probabilistic & Fuzzy Uncertainty Assessment", section_style))
        fuzzy_level = analysis_data.get("fuzzy_uncertainty_level", "Moderate Uncertainty")
        fuzzy_score = float(analysis_data.get("fuzzy_uncertainty_score", 0.5))
        needs_review = analysis_data.get("needs_expert_review", False)
        abstain = analysis_data.get("abstain_recommended", False)
        
        unc_rows = [
            [
                Paragraph("<b>Fuzzy Uncertainty Score:</b>", body_style),
                Paragraph(f"{fuzzy_score:.3f} ({fuzzy_level})", body_style),
                Paragraph("<b>Expert Review Flag:</b>", body_style),
                Paragraph("<font color='#dc2626'><b>MANDATORY</b></font>" if needs_review else "<font color='#16a34a'>Standard Follow-up</font>", body_style)
            ],
            [
                Paragraph("<b>Abstention Decision:</b>", body_style),
                Paragraph("<font color='#dc2626'><b>Abstain Recommended</b></font>" if abstain else "Confident to Proceed", body_style),
                Paragraph("<b>Epistemic (Model) Uncertainty:</b>", body_style),
                Paragraph(f"{analysis_data.get('epistemic_uncertainty', 0.0):.3f}", body_style)
            ]
        ]
        unc_table = Table(unc_rows, colWidths=[140, 130, 140, 130])
        unc_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#fef2f2") if needs_review else colors.HexColor("#f0fdf4")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#f87171") if needs_review else colors.HexColor("#86efac")),
            ('PADDING', (0,0), (-1,-1), 5),
        ]))
        elements.append(unc_table)
        elements.append(Spacer(1, 8))

        # Fuzzy explanation
        fuzzy_exp = analysis_data.get("fuzzy_reasoning_explanation", "")
        if fuzzy_exp:
            exp_p = Paragraph(f"<b>Fuzzy Reasoning Explanation:</b><br/>{fuzzy_exp.replace(chr(10), '<br/>')}", body_style)
            elements.append(exp_p)
            elements.append(Spacer(1, 8))

        # Section: Radiomic Features
        features = analysis_data.get("extracted_features", {})
        if features:
            elements.append(Paragraph("4. Segmented Radiomic & Quantitative Feature Profile", section_style))
            i_f = features.get("intensity_features", {})
            s_f = features.get("shape_features", {})
            t_f = features.get("texture_features", {})
            
            feat_table_data = [
                ["Feature Group", "Measurement", "Value", "Clinical Significance"],
                ["Morphometry", "Lesion Area", f"{s_f.get('area_pixels', 0)} pixels", "Volumetric burden indicator"],
                ["Morphometry", "Circularity", f"{s_f.get('circularity', 0.0):.2f}", "Sphericity (low = lobulated/infiltrative)"],
                ["Morphometry", "Boundary Irregularity", f"{s_f.get('boundary_irregularity', 0.0):.2f}", "Margin jaggedness index"],
                ["Intensity", "Mean Signal Intensity", f"{i_f.get('mean_intensity', 0.0):.1f}", "Tissue attenuation / spin density"],
                ["Intensity", "Signal Variance (Std)", f"{i_f.get('std_intensity', 0.0):.1f}", "Internal tissue heterogeneity"],
                ["GLCM Texture", "Contrast", f"{t_f.get('glcm_contrast', 0.0):.1f}", "Local intensity variations across voxels"],
                ["GLCM Texture", "Homogeneity", f"{t_f.get('glcm_homogeneity', 0.0):.3f}", "Structural uniformity of lesion matrix"],
            ]
            feat_table = Table(feat_table_data, colWidths=[100, 140, 110, 190])
            feat_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0f3d64")),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")),
                ('PADDING', (0,0), (-1,-1), 4),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
            ]))
            elements.append(feat_table)
            elements.append(Spacer(1, 10))

        # Decision Support Text
        elements.append(Paragraph("5. Clinical Decision Support Synthesis", section_style))
        dec_summary = analysis_data.get("decision_summary", "No clinical summary available.")
        elements.append(Paragraph(dec_summary.replace("\n", "<br/>"), body_style))
        elements.append(Spacer(1, 12))

        # Legal & Medical Disclaimer
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1"), spaceAfter=6))
        elements.append(Paragraph(
            "<b>IMPORTANT REGULATORY DISCLAIMER:</b> This report is generated by an academic artificial intelligence "
            "prototype ('Intelligent Medical Image-Based Tumor Detection and Uncertainty Assessment System') strictly for "
            "clinical research and investigational decision-support purposes. It is NOT FDA-approved or CE-certified for "
            "diagnostic use. All findings, probabilities, and uncertainty estimates must be evaluated by a qualified, board-certified radiologist.",
            disclaimer_style
        ))

        # Build PDF
        doc.build(elements)
        logger.info(f"Generated PDF report: {pdf_path}")
        return str(pdf_path)
