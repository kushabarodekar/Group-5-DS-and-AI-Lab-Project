# Milestone 6 Team Contribution Tracker

## AI-Powered Driver Wellness & Safety Monitoring System

**Milestone:** 6 — Deployment & Documentation  
**Deadline:** 13 August 2026  
**Project Stage:** Final Integration, Deployment, Documentation & Final Report

**Milestone 6 Lead:** Sohini

---

# 1. Milestone 6 Objective

Milestone 6 focuses on taking the five individually developed and evaluated modules from Milestones 1–5 and converting them into a **single integrated Driver Wellness & Safety Monitoring System**.

The milestone has two major goals:

## A. Deployment

- Integrate all five trained models into one end-to-end inference pipeline.
- Connect the individual module outputs to the Risk Fusion / Driver Wellness Score layer.
- Build a user-facing demo interface using **Gradio**.
- Deploy the integrated application using **Hugging Face Spaces**.
- Validate the deployed system using representative driver videos.
- Document deployment requirements, limitations, runtime behavior and known issues.

## B. Documentation

- Combine each member's work from **Milestones 1–5**.
- Add each member's **Milestone 6 contribution**.
- Prepare one **individual consolidated report per team member**.
- Combine the five individual reports into **one final consolidated project report**.
- Prepare the final presentation and submission artifacts.
- Maintain a complete contribution tracker showing individual responsibilities.

> **Note:** Hugging Face + Gradio deployment has been initiated. Ravina prepared the initial deployment documentation, and Sohini subsequently modified deployment parameters/configuration in the Hugging Face Space. Shiwani also attempted deployment using the same technical approach based on Ravina's initial deployment documentation. The current deployment is still facing ZeroGPU quota-exceeded issues during runtime inference, so further team support is required for deployment testing and stabilization.

---

# 2. Milestone 6 Requirements

The official Milestone 6 requirements are:

1. Deploy the model using an API, demo interface, or Hugging Face Spaces.
2. Prepare comprehensive documentation.
3. Finalize the project report.

For this project, these requirements are expanded into:

## 2.1 End-to-End Integration

- Integrate all five trained models.
- Standardize module inputs and outputs.
- Connect all modules to the Risk Fusion Engine.
- Generate the Driver Wellness Score.
- Generate the final risk category:
  - Safe
  - Caution
  - High Risk
- Generate the final Driver Safety Report.
- Handle module failures and invalid inputs safely.

## 2.2 Gradio Interface

- Create a video upload interface.
- Process uploaded driver videos.
- Display annotated output video.
- Display module-wise predictions.
- Display confidence scores.
- Display module risk scores.
- Display overall Driver Wellness Score.
- Display overall risk category.
- Display major safety warnings.

## 2.3 Hugging Face Deployment

- Create/configure Hugging Face Space.
- Prepare `app.py`, `wellness_core.py` and `README.md`
- Prepare `requirements.txt`.
- Configure model loading.
- Configure required dependencies.
- Test the application locally.
- Deploy to Hugging Face.
- Test the deployed application.
- Record deployment limitations.

## 2.4 Documentation

To prepare an **individual consolidated report** containing:

- Milestone 1 contribution
- Milestone 2 contribution
- Milestone 3 contribution
- Milestone 4 contribution
- Milestone 5 contribution
- Milestone 6 contribution
- Overall technical contribution
- Integration contribution
- Deployment contribution
- Artifacts/code produced
- Limitations
- Future improvements

The five individual reports are then combined into one:

**Final Consolidated Project Report**

---

# 3. Team Members and Module Ownership

| Team Member | Primary Module | Milestone 6 Primary Responsibility |
|---|---|---|
| **Kushagra** | Video-Based Fatigue Detection | Video fatigue module integration, inference validation and integration testing |
| **Shiwani** | Landmark-Based Fatigue Detection | Landmark fatigue module integration, feature/input validation and temporal output verification, deployment |
| **Shubham** | Driver Activity Classification | Driver Activity integration and module verification, deployment |
| **Sohini** | Seat Belt & Phone Usage Detection | Seatbelt/Phone Detection Module integration, detection validation, deployment support/coordination, documentation coordination, contribution tracking, final consolidated reports and final presentation preparation |
| **Ravina** | Smoking & Drinking Detection | Smoking/Drinking module integration, initial Hugging Face deployment documentation and deployment support |

> **Common responsibility:** Every team member is responsible for integrating, testing, documenting and validating their own module within the final end-to-end system.

---

# 4. Detailed Contribution Distribution

## 4.1 Kushagra — Video-Based Fatigue Detection

### Primary Responsibility

**Video-Based Fatigue Detection Integration**

### Milestone 6 Tasks

#### A. Model Integration

- Integrate the final Milestone-5 video fatigue checkpoint.
- Verify checkpoint loading in the integrated application.
- Connect the existing video preprocessing pipeline.
- Verify frame/sequence generation.
- Verify temporal/sliding-window inference.
- Ensure compatibility with the common video input format used by the integrated system and live video stream, for integrated module.

#### B. Output Standardization

Convert the module output into the common format:

```python
{
    "module": "video_fatigue",
    "prediction": "...",
    "confidence": 0.00,
    "risk_score": 0.00,
    "status": "OK"
}
```

#### C. Integration Testing

Test the module using:

- Normal/safe driving videos
- Drowsy/high-risk sequences
- Short videos
- Low-quality videos
- Videos with insufficient frames
- Invalid/incomplete video input

#### D. Risk Fusion Verification

- Verify that the module's risk score is correctly passed to the Risk Fusion Engine.
- Verify that the module contribution appears correctly in:
  - Module Summary
  - Risk Breakdown
  - Prediction Table
  - Timeline
  - Overall Driver Wellness Score

#### E. Documentation

Prepare the M6 section covering:

- Integration process
- Input/output format
- Final checkpoint
- Runtime behavior
- Integration testing
- Known limitations
- Future improvements

### Deliverables

- Integrated modules
- Test results
- Sample outputs
- M6 documentation
- Individual M6 report contribution

---

## 4.2 Shiwani — Landmark-Based Fatigue Detection

### Primary Responsibility

**Landmark-Based Fatigue Detection Integration and Validation**

### Milestone 6 Tasks

#### A. Model/Pipeline Integration

- Integrate the final landmark-based fatigue pipeline.
- Connect face landmark extraction with the final inference pipeline.
- Verify compatibility with uploaded videos.
- Verify frame-by-frame landmark processing.
- Verify temporal aggregation of fatigue-related features.

#### B. Feature Validation

Verify that the following features are correctly generated and passed to the model:

- Eye-related features
- Mouth-related features
- Head pose
- Facial landmarks
- Temporal fatigue indicators

Verify that invalid/missing landmark frames are handled safely.

#### C. Output Standardization

Convert the module output into:

```python
{
    "module": "landmark_fatigue",
    "prediction": "...",
    "confidence": 0.00,
    "risk_score": 0.00,
    "status": "OK"
}
```

#### D. Integration Testing

Test:

- Alert/awake driving
- Drowsy driving
- Different head poses
- Partial face visibility
- Poor lighting
- Landmark detection failures
- Short videos

#### E. Risk Fusion Verification

Verify that the landmark fatigue risk score is correctly incorporated into:

- Overall risk
- Module-wise risk
- Timeline
- Final Driver Wellness Score

Verify that temporal outputs are correctly generated, stabilized and passed to the integrated system.

#### F. Deployment Support

- Support the team in deploying the integrated application using the selected deployment platform.
- Assist with deployment testing and troubleshooting.
- Verify that the landmark-based fatigue module works correctly within the deployed application.
- Test landmark feature extraction and temporal inference using uploaded videos in the deployed environment.
- Document any deployment-specific issues related to MediaPipe, model loading, GPU/runtime availability or video processing.
- Coordinate with the deployment team to investigate and resolve deployment-related issues.

#### G. Documentation

Document:

- Final preprocessing pipeline
- Feature extraction
- Model integration
- Temporal processing
- Failure handling
- M6 evaluation/testing
- Limitations

### Deliverables

- Integrated landmark fatigue module
- Feature validation results
- Test samples
- Deployment support/testing
- M6 documentation
- Individual M6 report contribution

---

## 4.3 Shubham — Driver Activity Classification

### Primary Responsibility

**Driver Activity Integration**

### Milestone 6 Tasks

#### A. Driver Activity Integration

- Integrate the final Driver Activity Classification model.
- Verify model loading.
- Connect preprocessing and inference.
- Verify class mapping.
- Verify confidence calculation.
- Standardize output.

```python
{
    "module": "driver_activity",
    "prediction": "...",
    "confidence": 0.00,
    "risk_score": 0.00,
    "status": "OK"
}
```

#### B. Documentation

Document:

- Driver Activity integration
- Input/output format
- Final checkpoint
- Runtime behavior
- Integration testing
- Known limitations
- Future improvements


#### C. Deployment Support

- Support the team in deploying the integrated application using the selected deployment platform.
- Assist with deployment testing and troubleshooting.
- Verify that the landmark-based fatigue module works correctly within the deployed application.
- Test landmark feature extraction and temporal inference using uploaded videos in the deployed environment.
- Document any deployment-specific issues related to MediaPipe, model loading, GPU/runtime availability or video processing.
- Coordinate with the deployment team to investigate and resolve deployment-related issues.

### Deliverables

- Integrated Driver Activity module
- Test results
- M6 documentation
- Individual M6 report contribution
- Deployment support/testing

---

## 4.4 Sohini — Seat Belt & Phone Usage Detection

### Primary Responsibility

**Seatbelt/Phone Detection Module integration, detection validation, deployment support/coordination, documentation coordination, contribution tracking, final consolidated reports and final presentation preparation**

### Milestone 6 Tasks

#### A. Seatbelt/Phone Module integration

- Integrate the final Seat Belt & Phone Usage Detection model.
- Verify checkpoint loading.
- Verify YOLO inference configuration.
- Verify class mapping:

```python
{
    0: "Phone",
    1: "Seatbelt"
}
```

- Verify confidence thresholds.
- Verify NMS configuration.
- Verify temporal consensus/stabilization.
- Verify streaming inference.

#### B. Detection Validation

Specifically validate:

- Phone detection
- Seatbelt detection
- Phone Only
- Seatbelt Only
- Phone & Seatbelt
- No Detection

Investigate:
- false positives
- false negatives
- incorrect class predictions
- confidence-related issues
- temporal instability 
- detection inconsistencies observed during integration.

Document the observed issues, their causes where identified, and the corresponding fixes or mitigation strategies.

#### C. Annotated Output Verification and Contribution Tracking

Verify that:

- Bounding boxes appear correctly.
- Class labels are correct.
- Confidence values are displayed correctly.
- Boxes are temporally stable.
- Detection output corresponds to the final prediction.
- Annotated output is consistent with the module's standardized prediction.

#### D. Integration Output

Standardize output as:

```python
{
    "module": "seatbelt_phone",
    "prediction": "...",
    "confidence": 0.00,
    "risk_score": 0.00,
    "status": "OK"
}
```

Verify that the standardized output can be consumed correctly by the downstream Risk Fusion Engine and integrated application.

#### E. Risk Fusion Verification

Verify that:

```
Phone Only
        ↓
High Risk

Seatbelt Only
        ↓
Appropriate Seatbelt Risk

Phone & Seatbelt
        ↓
High Risk

No Detection
        ↓
Low/No Detection Risk
```

is correctly reflected by the Risk Fusion Engine according to the final project configuration.

#### F. Deployment Support and Coordination

- Support the team in validating the Seatbelt/Phone module within the Hugging Face deployment environment.
- Verify that the module can load its required model/checkpoint and configuration in the deployed environment.
- Assist in investigating runtime inference issues affecting the integrated application.
- Coordinate with Ravina regarding the initial Hugging Face deployment documentation and configuration.
- Document the current ZeroGPU quota limitation affecting runtime inference.
- Coordinate with other team members to obtain additional support required to complete deployment validation.

#### G. Final Consolidated Report

Coordinate the creation and consolidation of:

```
Member 1 Report
        +
Member 2 Report
        +
Member 3 Report
        +
Member 4 Report
        +
Member 5 Report
        +
Member 6 Report
        ↓
Final Consolidated Project Documentation
```

The final report should contain:

1. Problem Definition
2. Literature Review
3. Dataset Preparation
4. Model Architecture
5. Training
6. Evaluation
7. End-to-End Integration
8. Deployment
9. Gradio Interface
10. Hugging Face Deployment
11. System Architecture
12. Final Results
13. Error Analysis
14. Limitations
15. Future Work
16. Individual Contributions
17. References

#### H. Documentation Coordination

Prepare and coordinate:

- Final YOLO configuration
- Detection validation results
- Sample annotated outputs
- False-positive/false-negative observations
- Integration issues and fixes
- Known limitations
- Future improvements
- Collection and formatting of each member's M1-M6 report contribution
- Final consolidated project documentation
- Consolidation and updating of the M1–M5 technical report, non-technical report, user guide, developer guide and code documentation
- Incorporation of M6-specific updates into the above documents as the corresponding work is completed
- Coordination with each member to review and verify their respective module sections before final submission
- Cross-checking technical details across the final reports, presentation, user guide, developer guide and contribution tracker for consistency

#### I. Contribution Tracker

Maintain the final:

```text
M1 → Individual Contributions
M2 → Individual Contributions
M3 → Individual Contributions
M4 → Individual Contributions
M5 → Individual Contributions
M6 → Individual Contributions
```

tracker.

Ensure that every member's contribution is clearly recorded.

#### J. Final Presentation Preparation

The presentation should include:

- Project problem statement
- Objectives
- Proposed solution
- Overall methodology
- M1–M6 development journey
- Final system architecture
- Individual module architectures
- Dataset and preprocessing overview
- Model architectures
- Training and evaluation results
- End-to-end integration
- Risk Fusion Engine
- Application/interface workflow
- Deployment progress
- Deployment challenges and current status
- Final results
- Error analysis
- Limitations
- Future work
- Individual team contributions
- Key technical and project takeaways

Coordinate with each member to verify their respective module information, results and technical claims before including them in the final presentation.
Ensure that the final presentation is consistent with the final technical report, non-technical report, user guide, developer guide and contribution tracker.

### Deliverables

- Integrated YOLO module
- Detection validation results
- Deployment validation/support records
- Final consolidated reports
- M6 documentation
- Final presentation
- Contribution tracker
- Individual M1–M6 report contributions

---

## 4.5 Ravina — Smoking & Drinking Detection + Deployment Support

### Primary Responsibility

**Smoking/Drinking Integration + Initial Hugging Face Deployment Documentation and Deployment Support + Final Presentation Preparation**

### Milestone 6 Tasks

#### A. Smoking & Drinking Integration

- Integrate the final Smoking & Drinking Detection model.
- Verify model loading.
- Verify preprocessing.
- Verify class mapping.
- Verify confidence scores.
- Verify risk mapping.
- Standardize output.

```python
{
    "module": "smoking_drinking",
    "prediction": "...",
    "confidence": 0.00,
    "risk_score": 0.00,
    "status": "OK"
}
```

#### B. Module Testing

Test:

- Safe/normal driving
- Smoking
- Drinking
- Different camera conditions
- Occlusion
- Low-quality frames
- False-positive scenarios

Record relevant observations, issues and fixes identified during module integration and testing.

#### C. Risk Fusion Validation

Verify that Smoking & Drinking outputs are correctly reflected in:

- Module Summary
- Risk Breakdown
- Prediction Table
- Overall Wellness Score
- Final risk category

#### D. Initial Deployment Documentation and Hugging Face Deployment Support

- Prepare and maintain the initial Gradio/Hugging Face deployment documentation.
- Document the initial deployment procedure and configuration used for the Hugging Face Space.
- Provide deployment configuration notes and technical guidance to the team.
- Support Hugging Face deployment testing and troubleshooting.
- Record deployment steps and configuration changes.
- Capture relevant deployment screenshots and evidence.
- Maintain the deployment checklist.
- Test the Smoking/Drinking module in the deployed application where runtime availability permits.
- Assist other team members in understanding and following the initial deployment procedure.
- Coordinate with team members on subsequent deployment testing and configuration changes.

#### E. Individual Report Contribution

Prepare and submit Ravina's own:

- M6 report

Ensure that the module-specific technical information, results, challenges, solutions and M6 updates are provided to Sohini for documentation coordination and final consolidated report preparation.

### Deliverables

- Integrated Smoking/Drinking module
- Module testing results
- Initial Hugging Face deployment documentation
- Deployment support and troubleshooting records
- M6 report contribution
- Deployment-related screenshots/evidence where available

---

# 5. Common Responsibilities — All Team Members

Although each member has a primary responsibility, the following tasks are **shared by all five members**.

## 5.1 End-to-End Integration Testing

Every member should participate in testing the complete pipeline:

```text
Input Driver Video
        ↓
┌───────────────────────────────────────┐
│        Five Detection Modules         │
├───────────────────────────────────────┤
│ Video Fatigue                         │
│ Landmark Fatigue                      │
│ Driver Activity                       │
│ Smoking & Drinking                    │
│ Seat Belt & Phone                     │
└───────────────────────────────────────┘
        ↓
Risk Fusion Engine
        ↓
Driver Wellness Score
        ↓
Risk Category
        ↓
Gradio Interface
        ↓
Hugging Face Space
```

## 5.2 Input/Output Compatibility

Each member must verify:

- Input format
- Output format
- Class names
- Confidence values
- Risk scores
- Error states
- Runtime behavior

## 5.3 Failure Handling

The integrated application should not fail completely if one module encounters an issue.

Examples:

- Model loading failure
- Unsupported video
- Corrupted video
- Missing frames
- Detection failure
- Landmark failure
- Insufficient video duration

The failure should be reported clearly.

## 5.4 Final System Validation

All members should participate in testing representative videos covering combinations such as:

| Scenario | Expected Testing |
|---|---|
| Normal driving | All modules |
| Drowsy driving | Fatigue modules |
| Phone usage | Seat Belt & Phone |
| No seatbelt | Seat Belt & Phone |
| Smoking | Smoking Detection |
| Drinking | Drinking Detection |
| Distracted activity | Driver Activity |
| Multiple simultaneous risks | Full Risk Fusion |
| Poor-quality video | Robustness |
| Short video | Input validation |

---

# 6. Risk Fusion & Wellness Score — Shared Integration Task

The Risk Fusion Engine is a **team-level component**.

The team will verify that all module outputs are correctly combined.

Each module provides:

```text
Prediction
Confidence
Risk Score
Status
```

The Risk Fusion Engine combines these values into:

```text
Module Risks
     ↓
Weighted Risk Fusion
     ↓
Overall Driver Wellness Score
     ↓
Risk Category
```

The final system should clearly show:

- Individual module risk
- Individual module confidence
- Weighted contribution
- Overall wellness score
- Overall risk category

The final risk categories should follow the project's established configuration.

---

# 7. Gradio Interface Work Distribution

## Lead: Shubham

## Supporting Members: All

The interface should contain at minimum:

## Input Section

- Video upload
- Analyze button
- Input validation

## Video Output

- Annotated driver video

## Module Results

| Module | Prediction | Confidence | Risk |
|---|---|---:|---:|
| Video Fatigue | ... | ... | ... |
| Landmark Fatigue | ... | ... | ... |
| Driver Activity | ... | ... | ... |
| Smoking & Drinking | ... | ... | ... |
| Seat Belt & Phone | ... | ... | ... |

## Overall Result

```text
Driver Wellness Score: XX / 100

Risk Level: Safe / Caution / High Risk
```

## Safety Warnings

Display the most important detected risks.

---

# 8. Deployment Work Distribution

### Deployment Status

**Current status:** Lightning AI deployment has been initiated and the application/models have been pushed. Further team members assisted with deployment testing, optimization and stabilization.

### Tasks

#### Ravina

- Prepare the initial Hugging Face deployment documentation.
- Record deployment steps and configuration.
- Support Hugging Face configuration and deployment testing.
- Maintain deployment checklist.

#### Sohini

- Verify YOLO model loading on deployment.
- Verify annotated video output.
- Verify Seatbelt/Phone prediction display.

#### Shiwani

- Deployment support
- Help investigate deployment/runtime issues.
- Track deployment limitations and final deployment test status.

#### Kushagra

- Verify Video Fatigue model loading on deployment when runtime testing is available.
- Test video fatigue output.
- Test the integrated application locally.
- Support deployment-specific model validation.
- Modify deployment parameters/configuration on other platforms

#### Shubham

- Modify deployment parameters/configuration on other platforms also
- Continue responsibility for Driver Activity module integration and verification.

---

# 9. Documentation Work Distribution

Documentation will be handled at **two levels**.

## Level 1 — Individual Reports

Each member prepares their own consolidated report:

```text
M1 Contribution
        ↓
M2 Contribution
        ↓
M3 Contribution
        ↓
M4 Contribution
        ↓
M5 Contribution
        ↓
M6 Contribution
        ↓
Individual Technical Contribution
```

Each report should describe the member's actual work rather than simply reproducing the milestone requirements.

### Individual M6 Section

Each member should include:

- M6 objectives
- Their assigned M6 responsibilities
- Integration work
- Testing performed
- Problems encountered
- Solutions implemented
- Final results
- Deployment involvement
- Documentation contribution
- Limitations
- Future improvements
- Artifacts/code produced

---

# 10. Final Consolidated Report

The individual reports will be combined into a single final project report.

## Proposed Structure

```text
1. Abstract
2. Introduction
3. Problem Statement
4. Objectives and Scope
5. Literature Review
6. Existing Solutions
7. Dataset Description
8. Dataset Preparation and Quality Analysis
9. System Architecture
10. Individual Model Architectures
11. End-to-End Pipeline
12. Model Training
13. Model Evaluation
14. Error Analysis
15. Risk Fusion Framework
16. Driver Wellness Score
17. Integrated System
18. Gradio Interface
19. Hugging Face Deployment
20. Deployment Testing
21. Final Results
22. Limitations
23. Challenges and Solutions
24. Individual Contributions
25. Future Work
26. Conclusion
27. References
```

---

# 11. Final Report Consolidation Process

The final documentation process will follow:

```text
Member 1 M1–M6
       ↓
Individual Report 1

Member 2 M1–M6
       ↓
Individual Report 2

Member 3 M1–M6
       ↓
Individual Report 3

Member 4 M1–M6
       ↓
Individual Report 4

Member 5 M1–M6
       ↓
Individual Report 5

       ↓↓↓

Final Consolidation

       ↓

Single Final Project Report
```

### Important

The final consolidated report should **not simply concatenate five reports**.

The team should:

- Remove duplicated explanations.
- Maintain one consistent project narrative.
- Merge common sections.
- Preserve member-specific technical contributions.
- Keep module-specific implementation details.
- Maintain consistent terminology.
- Cross-reference the five modules.
- Include the integrated architecture.
- Include deployment details.
- Include final system results.

---

# 12. Final Presentation

## Shared Responsibility

All members should contribute to the final presentation.

## Suggested Presentation Structure

### Slide 1 — Title

AI-Powered Driver Wellness & Safety Monitoring System

### Slide 2 — Problem Statement

### Slide 3 — Objectives

### Slide 4 — Existing Solutions & Motivation

### Slide 5 — Dataset

### Slide 6 — Overall System Architecture

### Slide 7 — Five Detection Modules

### Slide 8 — Model Architectures

### Slide 9 — Training

### Slide 10 — Evaluation Results

### Slide 11 — Error Analysis

### Slide 12 — Risk Fusion

### Slide 13 — Driver Wellness Score

### Slide 14 — Integrated Pipeline

### Slide 15 — Gradio Interface

### Slide 16 — Hugging Face Deployment

### Slide 17 — Sample Outputs

### Slide 18 — Limitations

### Slide 19 — Individual Contributions

### Slide 20 — Conclusion & Future Work

---

# 13. M6 Testing Checklist

## Model Integration

- [ ] Video Fatigue model integrated
- [ ] Landmark Fatigue model integrated
- [ ] Driver Activity model integrated
- [ ] Smoking/Drinking model integrated
- [ ] Seat Belt/Phone model integrated

## Output Validation

- [ ] Predictions correct
- [ ] Confidence values correct
- [ ] Risk scores correct
- [ ] Class labels correct
- [ ] Module status correctly reported

## Risk Fusion

- [ ] All five module outputs received
- [ ] Risk weights verified
- [ ] Overall score calculated
- [ ] Risk category generated
- [ ] Timeline generated correctly

## Interface

- [ ] Video upload works
- [ ] Input validation works
- [ ] Analysis button works
- [ ] Annotated video generated
- [ ] Module summary displayed
- [ ] Overall score displayed
- [ ] Risk category displayed
- [ ] Safety warnings displayed

## Deployment

- [ ] `app.py` completed
- [ ] `requirements.txt` completed
- [ ] Local testing completed
- [ ] Hugging Face Space created
- [ ] Models load successfully
- [ ] Inference works remotely
- [ ] Output video works remotely
- [ ] Deployment limitations documented

## Documentation

- [ ] Individual M1–M6 report — Kushagra
- [ ] Individual M1–M6 report — Shiwani
- [ ] Individual M1–M6 report — Shubham
- [ ] Individual M1–M6 report — Sohini
- [ ] Individual M1–M6 report — Ravina
- [ ] Contribution tracker updated
- [ ] Final report consolidated
- [ ] References checked
- [ ] Screenshots added
- [ ] Final presentation completed
- [ ] Submission checklist completed

---

# 14. Final Deliverables

The following artifacts should be ready for Milestone 6 submission:

### Code

- Integrated inference pipeline
- Individual model adapters/modules
- Risk Fusion Engine
- Gradio application
- Deployment configuration
- Supporting utility scripts

### Models

- Final trained checkpoints
- Required model configuration files

### Documentation

- Final consolidated project report
- Updated contribution tracker
- Deployment documentation

### Presentation

- Final milestone presentation slides
- Architecture diagrams
- Results visualizations
- Sample predictions
- Deployment screenshots

### Deployment

- Hugging Face Space
- Gradio interface
- Tested deployed application

---

# 15. Definition of Done

Milestone 6 will be considered complete when:

- [ ] All five models are integrated into one pipeline.
- [ ] All module outputs follow a consistent format.
- [ ] Risk Fusion successfully combines all module outputs.
- [ ] Driver Wellness Score is generated.
- [ ] Final risk category is generated.
- [ ] Annotated video output is available.
- [ ] Gradio interface is functional.
- [ ] Application is deployed to Hugging Face Spaces.
- [ ] Deployed application has been tested.
- [ ] Known issues and limitations are documented.
- [ ] Each member has completed their M1–M6 individual report.
- [ ] Contribution tracker is updated.
- [ ] Individual reports are consolidated.
- [ ] Final project report is completed.
- [ ] Final presentation is completed.
- [ ] All code, reports and deployment artifacts are ready for submission.

---

# 16. Milestone 6 Responsibility Summary

**Milestone 6 Lead:** Sohini

| Member | Module Integration | Contribution | Signature |
|---|---|---|---|
| **Kushagra** | Video-Based Fatigue Detection | Video fatigue integration, inference validation and end-to-end integration testing | ____________________ |
| **Shiwani** | Landmark-Based Fatigue Detection | Landmark integration, feature/input validation and temporal output verification | ____________________ |
| **Shubham** | Driver Activity Classification | Driver Activity integration, Gradio interface and Hugging Face deployment | ____________________ |
| **Sohini** | Seat Belt & Phone Usage Detection | Seatbelt/Phone Detection Module integration, documentation coordination, contribution tracking and final consolidated report | ____________________ |
| **Ravina** | Smoking & Drinking Detection | Smoking/Drinking integration, Hugging Face deployment along with Shubham and final presentation preparation | ____________________ |
