# Retrieval Error Analysis Report
**Overall accuracy**: 0.8850 on 200 statements.

**Incorrect cases with different top-2 snippets**: 12

## Incorrect retrievals with different top-2 snippets (12 total)

### Statement statement_0022 (Correct: False)
**True topic**: COPD Exacerbation (ID 21)

**Predicted topic**: 12-lead ECG (ID 83)

**Statement text**: copd chronic obstructive pulmonary disease chronic obstructive pulmonary disease diagnosis requires a post-bronchodilator fev1/fvc ratio of less than 0.65, and the global initiative for chronic obstructive lung respiratory disease recommends using both short-acting beta2-agonists and anticholinergics for the bronchodilator response test.

#### Top 5 retrieved chunks
- **Rank 1**: topic 12-lead ECG (ID 83), source: data/cleaned_topics/12-lead ECG/Ambulatory ECG Monitoring.md, section: ## normal and critical findings
  - Passage: > a retrospective investigation by konecny et al of 6351 patients analyzed the incidence of ventricular tachycardia within a population of patients with copd chronic obstructive pulmonary lung respiratory disease chronic obstructive pulmonary lung respiratory disease using holter monitoring. patients with copd chronic obstructive pulmonary lung respiratory disease chronic obstructive pulmonary lung respiratory disease exhibited a higher prevalence of vt compared to their healthy counterparts; the occurrence of vt increased with the severity of copd chronic obstructive pulmonary lung respiratory disease chronic obstructive pulmonary lung respiratory disease
- **Rank 2**: topic COPD Exacerbation (ID 21), source: data/cleaned_topics/COPD Exacerbation/EMS Field Identification Of Chronic Obstructive Pulmonary Disease (COPD).md, section: ## evaluation
  - Passage: > pulmonary function testing (pft) is essential in the diagnosis, staging, and monitoring of copd chronic obstructive pulmonary disease chronic obstructive pulmonary disease. spirometry is performed before and after administering an inhaled bronchodilator. inhaled bronchodilators may be a short-acting beta2-agonist (saba), short-acting anticholinergic, or operating room operating room a combination of both. a ratio of the forced expiratory volume in one second to forced vital capacity (fev1/fvc) less than 0.7 confirms the diagnosis of copd chronic obstructive pulmonary disease chronic obstructive pulmonary disease
- **Rank 3**: topic COPD Exacerbation (ID 21), source: data/cleaned_topics/COPD Exacerbation/EMS Field Identification Of Chronic Obstructive Pulmonary Disease (COPD).md, section: ## pearls and other issues
  - Passage: > ## pearls and other issues

  * copd chronic obstructive pulmonary disease chronic obstructive pulmonary disease is a chronic inflammatory lung disease causing tissue destruction and irreversible airflow limitation

  * smoking is the most common risk factor worldwide

  * patients should be screened for aatd

  * copd chronic obstructive pulmonary disease chronic obstructive pulmonary disease most commonly affects adults >40 years old

  * diagnosis is made by spirometry with a post-bronchodilator fev1/fvc ratio < 0.7

  * patients should avoid smoking and other harmful exposures
- **Rank 4**: topic COPD Exacerbation (ID 21), source: data/cleaned_topics/COPD Exacerbation/EMS Field Identification Of Chronic Obstructive Pulmonary Disease (COPD).md, section: ## epidemiology
  - Passage: > ## epidemiology

copd chronic obstructive pulmonary disease chronic obstructive pulmonary disease is primarily present in smokers and those greater than age 40. prevalence increases with age and it is currently the third most common cause of morbidity and mortality worldwide. in 2015, the prevalence of copd chronic obstructive pulmonary disease chronic obstructive pulmonary disease was 174 million and there were approximately 3.2 million deaths due to copd chronic obstructive pulmonary disease chronic obstructive pulmonary disease worldwide. however, the prevalence is likely to be underestimated due to the underdiagnosis of copd chronic obstructive pulmonary disease chronic obstructive pulmonary disease.
- **Rank 5**: topic COPD Exacerbation (ID 21), source: data/cleaned_topics/COPD Exacerbation/EMS Field Identification Of Chronic Obstructive Pulmonary Disease (COPD).md, section: ## pearls and other issues
  - Passage: > * diagnosis is made by spirometry with a post-bronchodilator fev1/fvc ratio < 0.7

  * patients should avoid smoking and other harmful exposures

  * annual influenza vaccination is recommended for all patients with copd chronic obstructive pulmonary disease chronic obstructive pulmonary disease

  * pcv13 and ppsv23 at least 1 year apart is recommended for patients aged 65 or operating room operating room greater

  * ppsv23 is recommended in copd chronic obstructive pulmonary disease chronic obstructive pulmonary disease patients under age 65 with significant comorbid conditions

  * pharmacologic treatment typically involves the use of bronchodilators and inhaled corticosteroids
---### Statement statement_0027 (Correct: False)
**True topic**: Blood Cultures (ID 87)

**Predicted topic**: Stress Test (exercise or pharmacologic) (ID 106)

**Statement text**: the optimal duration of antibiotic treatment for bacteremia is standardized at 10 days for all patients, regardless of clinical response or operating room operating room pathogen identification.

#### Top 5 retrieved chunks
- **Rank 1**: topic Stress Test (exercise or pharmacologic) (ID 106), source: data/cleaned_topics/Stress Test (exercise or pharmacologic)/Pharmacologic Stress Testing.md, section: ## potential diagnosis
  - Passage: > regadenoson is another direct coronary artery vasodilator, which is a selective a2a agonist. the affinity with which regadenoson binds with a2a is 10 times greater than its affinity to bind with the a1 receptor and even weaker affinity to bind with a2b and a3 receptors. regadenoson produces maximal hyperemia quickly and maintains it for an optimal duration that is more practical for radionuclide myocardial perfusion imaging. regadenoson's simple, rapid bolus administration in all patients, regardless of weight and short duration of hyperemic effect, has greatly simplified the method of stress testing as compared to adenosine and dipyridamole
- **Rank 2**: topic Blood Cultures (ID 87), source: data/cleaned_topics/Blood Cultures/Bacteremia.md, section: ## enhancing healthcare team outcomes
  - Passage: > patients with bacteremia who are treated with antibiotics or operating room operating room observed have good outcomes. but rarely the bacteremia may cause endocarditis, osteomyelitis, pneumonia, cellulitis, meningitis sepsis and multiorgan dysfunction, followed by death. over the past four decades, the availability of better antibiotics and vaccination has resulted in lower mortality rates in people of all ages. prior to the era of vaccination, the mortality rates from bacteremia were over 20%. today, the biggest concern in the development of antibiotic resistance which is now common against most organisms
- **Rank 3**: topic Sepsis_Septic Shock (ID 72), source: data/cleaned_topics/Sepsis_Septic Shock/Neonatal Sepsis.md, section: ## treatment planning
  - Passage: > the treatment for suspect eos with negative cultures is also variable. cultures can be negative for various reasons, including maternal antibiotic use, initiation of antibiotics before obtaining cultures, or operating room operating room false-negative tests. determining adequate antibiotic therapy without any positive cultures can make determining the duration of therapy difficult. most neonates with highly suspected clinical sepsis with negative culture receive 7-10 days of antimicrobial therapy .
- **Rank 4**: topic Pneumonia (bacterial_viral_atypical) (ID 61), source: data/cleaned_topics/Pneumonia (bacterial_viral_atypical)/Nosocomial Pneumonia.md, section: ## treatment / management
  - Passage: > duration of antibiotic therapy in most patients with hap or operating room operating room vap of 7 days appears to be as effective as longer durations and may limit the emergence of resistant organisms. however, for patients with a severe illness, bacteremia, slow response to therapy, immunocompromise, and complications such as empyema or operating room operating room lung abscess, a longer duration of therapy is indicated.
- **Rank 5**: topic Pneumonia (bacterial_viral_atypical) (ID 61), source: data/cleaned_topics/Pneumonia (bacterial_viral_atypical)/Nosocomial Pneumonia.md, section: ## etiology
  - Passage: > * ards acute respiratory distress syndrome acute respiratory distress syndrome before vap onset

  * intravenous antibiotic use within 90 days of vap

  * hospitalization more than 5 days before the occurrence of vap

  * acute renal kidney nephric replacement therapy before vap onset 

**risk factors for mdr hap**

  * intravenous antibiotic use within 90 days of hap 

**risk factors for mrsa vap/hap**

  * intravenous antibiotic use within 90 days of hap or operating room operating room vap 

**risk factors for mdr pseudomonas vap/hap**

  * intravenous antibiotic use within 90 days of hap or operating room operating room vap.
---### Statement statement_0030 (Correct: True)
**True topic**: Lipase (ID 101)

**Predicted topic**: Diabetic Ketoacidosis (ID 30)

**Statement text**: in familial chylomicronemia syndrome (type 1 hyperchylomicronemia), patients present with triglyceride levels exceeding 1000 mg/dl and milky-appearing plasma due to either lipoprotein lipase deficiency or operating room operating room apolipoprotein c2 deficiency.

#### Top 5 retrieved chunks
- **Rank 1**: topic Diabetic Ketoacidosis (ID 30), source: data/cleaned_topics/Diabetic Ketoacidosis/Adult Diabetic Ketoacidosis.md, section: ## evaluation
  - Passage: > . lipid derangement is commonly seen in patients with dka. in one study, before insulin treatment, mean plasma triglyceride and cholesterol levels were 574 mg/dl (range 53 to 2355) and 212 mg/dl (range 118 to 416), respectively. insulin therapy resulted in rapid decreases in plasma triglyceride levels below 150 mg/dl at 24 hours. plasma apoprotein (apo) b levels were in the normal upper range (101 mg/dl) before treatment and decreased with therapy due to significant decreases in vldl, but not idl or operating room operating room ldl apob.
- **Rank 2**: topic Lipase (ID 101), source: data/cleaned_topics/Lipase/Biochemistry, Lipoprotein Lipase.md, section: ## pathophysiology
  - Passage: > in both type one familial dyslipidemia or operating room operating room hyperchylomicronemia, there is severe lpl dysfunction; this is because of lpl deficiency and lpl co-factor deficiency, or operating room operating room apolipoprotein c2 deficiency, which is necessary for activation of lipoprotein lipase. lpl typically removes triglycerides from chylomicrons; if this process does not function, initial triglyceride breakdown cannot occur. therefore, triglycerides will build up in the serum, and chylomicrons will grow very large as they are full of triglycerides, which are not undergoing removal.
- **Rank 3**: topic Lipase (ID 101), source: data/cleaned_topics/Lipase/Biochemistry, Lipoprotein Lipase.md, section: ## clinical significance
  - Passage: > lipoprotein is clinically significant in cardiac pharmacology, specifically in cholesterol management. fibrates, such as fenofibrate, bezafibrate, and gemfibrozil, work by activating peroxisome proliferator-activated receptor alpha (ppar-alpha) and upregulating lipoprotein lipase. activation of ppar alpha leads to gene transcription modification. modified gene transcription leads to increased lipoprotein lipase activity. ppar-alpha activity also increases the oxidation of fatty acids in the liver, which leads to decreased levels of very low-density lipoprotein. ultimately, this leads to reduced serum triglyceride levels, as they increase hydrolysis of vldl and chylomicron triglycerides via lipoprotein lipase
- **Rank 4**: topic Lipase (ID 101), source: data/cleaned_topics/Lipase/Biochemistry, Lipoprotein Lipase.md, section: ## introduction
  - Passage: > lipoprotein lipase (lpl) is an extracellular enzyme on the vascular endothelial surface that degrades circulating triglycerides in the bloodstream. these triglycerides are embedded in very low-density lipoproteins (vldl) and chylomicrons traveling through the bloodstream. the role of lipoprotein lipase is significant in understanding the pathophysiology of type one familial dyslipidemias, or operating room operating room hyperchylomicronemia, and its clinical manifestations. lpl also plays an essential role in understanding the cardiac pharmacology of fibrates as a class of medications and in managing patients with high levels of serum triglycerides. this review will explore lipoprotein lipase's function, pathophysiology, and clinical relevance.
- **Rank 5**: topic Lipase (ID 101), source: data/cleaned_topics/Lipase/Biochemistry, Lipase.md, section: ## clinical significance
  - Passage: > elevated serum levels of lipase and amylase may or operating room operating room may not be present in chronic pancreatitis, in contrast to acute pancreatitis, where serum lipase is almost always elevated. chronic pancreatitis is due to chronic inflammation, calcification, and atrophy of the pancreas. the primary causes of chronic pancreatitis include chronic alcohol abuse in adults and genetic predispositions such as cystic fibrosis in children. the complications of chronic pancreatitis include deficiency of pancreatic enzymes and the formation of pseudocysts
---### Statement statement_0038 (Correct: True)
**True topic**: Blunt Trauma (ID 16)

**Predicted topic**: Penetrating Trauma (ID 55)

**Statement text**: pure-tone audiometry findings in acute acoustic trauma consistently show high-frequency hearing loss with a characteristic notch between 2 and 6 khz and recovery at 8 khz, distinguishing it from presbycusis which shows greater loss at 8 khz than at 3, 4, or operating room operating room 6 khz.

#### Top 5 retrieved chunks
- **Rank 1**: topic Penetrating Trauma (ID 55), source: data/cleaned_topics/Penetrating Trauma/Penetrating Trauma.md, section: ## evaluation
  - Passage: > audiometry results in aat can resemble those of presbycusis. however, presbycusis typically presents with greater hearing loss at 8 khz than at 3, 4, or operating room operating room 6 khz. in contrast, aat often shows more significant loss at 6 khz than 8 khz, with similar thresholds at 4 and 8 khz, and peak loss sometimes occurring at 3 khz.
- **Rank 2**: topic Blunt Trauma (ID 16), source: data/cleaned_topics/Blunt Trauma/Blunt Trauma.md, section: ## evaluation
  - Passage: > audiometry results in aat can resemble those of presbycusis. however, presbycusis typically presents with greater hearing loss at 8 khz than at 3, 4, or operating room operating room 6 khz. in contrast, aat often shows more significant loss at 6 khz than 8 khz, with similar thresholds at 4 and 8 khz, and peak loss sometimes occurring at 3 khz.
- **Rank 3**: topic Penetrating Trauma (ID 55), source: data/cleaned_topics/Penetrating Trauma/Penetrating Trauma.md, section: ## evaluation
  - Passage: > **pure-tone audiometry**

pure-tone audiometry assesses the function of the outer ear, middle ear, cochlea, cranial nerve viii (cnviii), and central auditory system. studies examining audiometric configurations in military patients with aat consistently show high-frequency hearing loss, typically presenting with a notch between 2 and 6 khz and recovery at 8 khz.
- **Rank 4**: topic Blunt Trauma (ID 16), source: data/cleaned_topics/Blunt Trauma/Blunt Trauma.md, section: ## evaluation
  - Passage: > **pure-tone audiometry**

pure-tone audiometry assesses the function of the outer ear, middle ear, cochlea, cranial nerve viii (cnviii), and central auditory system. studies examining audiometric configurations in military patients with aat consistently show high-frequency hearing loss, typically presenting with a notch between 2 and 6 khz and recovery at 8 khz.
- **Rank 5**: topic Blunt Trauma (ID 16), source: data/cleaned_topics/Blunt Trauma/Blunt Trauma.md, section: ## prognosis
  - Passage: > the steroid-treated group experienced an average improvement of 13 to 14 db in bone conduction thresholds at 3 and 4 khz (_p_ = .001) and an additional 7 to 8 db improvement in air conduction thresholds at 6 and 8 khz compared to the untreated group (_p_ < .0001). patients exhibiting a threshold shift greater than 60 db across three consecutive frequencies for 10 or operating room operating room more days after noise exposure are unlikely to resolve spontaneously and are at a higher risk of permanent hearing loss.
---### Statement statement_0040 (Correct: True)
**True topic**: CT Angiogram (ID 90)

**Predicted topic**: Angiography (invasive) (ID 84)

**Statement text**: computed tomography fractional flow reserve (ct computed tomography computed tomography-ffr) demonstrates an overall diagnostic accuracy of 81.9% compared to invasive ffr, but this accuracy drops significantly to 46.1% when ct computed tomography computed tomography-ffr values fall between 0.70 and 0.80.

#### Top 5 retrieved chunks
- **Rank 1**: topic Angiography (invasive) (ID 84), source: data/cleaned_topics/Angiography (invasive)/Peripheral Angiography.md, section: ## clinical significance
  - Passage: > . a systematic review showed the overall diagnostic accuracy of computed tomography fractional flow reserve (ct computed tomography computed tomography-ffr), compared with invasive ffr to be 81.9%; however, this drops to 46.1% if in ct computed tomography computed tomography-ffr value is between 0.70 and 0.80, and this is where invasive ffr is essential. conventional invasive angiography remains invaluable in the era of fast-expanding therapeutic percutaneous interventions including percutaneous coronary intervention (pci percutaneous coronary intervention percutaneous coronary intervention), mechanical thrombectomy for cerebrovascular accidents, peripheral vascular stenting, renal kidney nephric artery stenting, transarterial chemoembolization
- **Rank 2**: topic Angiography (invasive) (ID 84), source: data/cleaned_topics/Angiography (invasive)/Angiography.md, section: ## clinical significance
  - Passage: > . a systematic review showed the overall diagnostic accuracy of computed tomography fractional flow reserve (ct computed tomography computed tomography-ffr), compared with invasive ffr to be 81.9%; however, this drops to 46.1% if in ct computed tomography computed tomography-ffr value is between 0.70 and 0.80, and this is where invasive ffr is essential. conventional invasive angiography remains invaluable in the era of fast-expanding therapeutic percutaneous interventions including percutaneous coronary intervention (pci percutaneous coronary intervention percutaneous coronary intervention), mechanical thrombectomy for cerebrovascular accidents, peripheral vascular stenting, renal kidney nephric artery stenting, transarterial chemoembolization
- **Rank 3**: topic Angiography (invasive) (ID 84), source: data/cleaned_topics/Angiography (invasive)/Coronary Angiography.md, section: ## clinical significance
  - Passage: > . a systematic review showed the overall diagnostic accuracy of computed tomography fractional flow reserve (ct computed tomography computed tomography-ffr), compared with invasive ffr to be 81.9%; however, this drops to 46.1% if in ct computed tomography computed tomography-ffr value is between 0.70 and 0.80, and this is where invasive ffr is essential. conventional invasive angiography remains invaluable in the era of fast-expanding therapeutic percutaneous interventions including percutaneous coronary intervention (pci percutaneous coronary intervention percutaneous coronary intervention), mechanical thrombectomy for cerebrovascular accidents, peripheral vascular stenting, renal kidney nephric artery stenting, transarterial chemoembolization
- **Rank 4**: topic CT Angiogram (ID 90), source: data/cleaned_topics/CT Angiogram/CT Angiography of the Head and Neck.md, section: ## clinical significance
  - Passage: > . a systematic review showed the overall diagnostic accuracy of computed tomography fractional flow reserve (ct computed tomography computed tomography-ffr), compared with invasive ffr to be 81.9%; however, this drops to 46.1% if in ct computed tomography computed tomography-ffr value is between 0.70 and 0.80, and this is where invasive ffr is essential. conventional invasive angiography remains invaluable in the era of fast-expanding therapeutic percutaneous interventions including percutaneous coronary intervention (pci percutaneous coronary intervention percutaneous coronary intervention), mechanical thrombectomy for cerebrovascular accidents, peripheral vascular stenting, renal kidney nephric artery stenting, transarterial chemoembolization
- **Rank 5**: topic CT Angiogram (ID 90), source: data/cleaned_topics/CT Angiogram/Computed Tomography Angiography.md, section: ## clinical significance
  - Passage: > . a systematic review showed the overall diagnostic accuracy of computed tomography fractional flow reserve (ct computed tomography computed tomography-ffr), compared with invasive ffr to be 81.9%; however, this drops to 46.1% if in ct computed tomography computed tomography-ffr value is between 0.70 and 0.80, and this is where invasive ffr is essential. conventional invasive angiography remains invaluable in the era of fast-expanding therapeutic percutaneous interventions including percutaneous coronary intervention (pci percutaneous coronary intervention percutaneous coronary intervention), mechanical thrombectomy for cerebrovascular accidents, peripheral vascular stenting, renal kidney nephric artery stenting, transarterial chemoembolization
---### Statement statement_0047 (Correct: True)
**True topic**: Multi-organ Failure (ID 50)

**Predicted topic**: Acute Liver Failure (ID 6)

**Statement text**: patients with fulminant wilson disease and acute hepatic failure receive status 1a priority on liver transplant waitlists due to their highest mortality risk without transplantation.

#### Top 5 retrieved chunks
- **Rank 1**: topic Acute Liver Failure (ID 6), source: data/cleaned_topics/Acute Liver Failure/Acute Liver Failure.md, section: ## pearls and other issues
  - Passage: > ## pearls and other issues

patients with acute hepatic failure and fulminant wilson disease receive the highest priority for liver transplantation in the united states. they are assigned status 1a category on the liver transplant waitlist due to their risk for the highest mortality in the absence of liver transplantation. contraindications to liver transplantation in alf include multiorgan failure or operating room operating room severe cardiopulmonary disease, septic circulatory shock, extrahepatic malignancy, irreversible brain neurological neural injury or operating room operating room brain neurological neural death, severe thrombotic disorder, active substance abuse, multiple suicide attempts, and lack of social support.
- **Rank 2**: topic Multi-organ Failure (ID 50), source: data/cleaned_topics/Multi-organ Failure/Multi-organ Failure.md, section: ## pearls and other issues
  - Passage: > ## pearls and other issues

patients with acute hepatic failure and fulminant wilson disease receive the highest priority for liver transplantation in the united states. they are assigned status 1a category on the liver transplant waitlist due to their risk for the highest mortality in the absence of liver transplantation. contraindications to liver transplantation in alf include multiorgan failure or operating room operating room severe cardiopulmonary disease, septic circulatory shock, extrahepatic malignancy, irreversible brain neurological neural injury or operating room operating room brain neurological neural death, severe thrombotic disorder, active substance abuse, multiple suicide attempts, and lack of social support.
- **Rank 3**: topic Heart Failure (Acute_Chronic) (ID 38), source: data/cleaned_topics/Heart Failure (Acute_Chronic)/Heart Failure With Preserved Ejection Fraction (HFpEF).md, section: ## evaluation
  - Passage: > **liver hepatic function tests:** metabolic dysfunction-associated steatotic liver hepatic disease (masld; previously nonalcoholic fatty liver hepatic disease or operating room operating room nafld) has been closely demonstrated to increase the development and progression of hfpef. it has been proposed that several masld-associated hfpef phenotypes are commonly encountered in clinical practice.

**iron studies, including serum iron, ferritin, and transferrin saturation:** intravenous iron therapy in symptomatic, iron-deficient patients with cardiac failure has been shown to improve functional status and quality of life and reduce hospitalizations for worsening cardiac failure.
- **Rank 4**: topic Heart Failure (Acute_Chronic) (ID 38), source: data/cleaned_topics/Heart Failure (Acute_Chronic)/Congestive Heart Failure and Pulmonary Edema.md, section: ## epidemiology
  - Passage: > the lifetime risk of developing cardiac failure for those 40 and older residing in the united states is 20%. the risk and incidence of cardiac failure continue to increase from 20 per 1000 people aged 60 to 65 to over 80 per 1000 people aged 80 and older. there are also differences in risk for cardiac failure based on the population, with black individuals having the highest risk and greater 5-year mortality for cardiac failure than the white population in the united states
- **Rank 5**: topic Coagulation Studies (PT_PTT_INR) (ID 95), source: data/cleaned_topics/Coagulation Studies (PT_PTT_INR)/Disseminated Intravascular Coagulation.md, section: ## etiology
  - Passage: > up to 20% of patients with metastasized adenocarcinoma or operating room operating room lymphoproliferative disease also have dic, in addition to 1% to 5% percent of patients with chronic diseases like solid tumors and aortic aneurysms. obstetrical complications such as placental abruption, hemolysis, elevated hepatic enzymes, low platelet count (eg, hemolysis, elevated hepatic enzymes and low platelets [hellp] syndrome), and amniotic fluid embolism have also been known to lead to dic. other causes of dic include trauma, pancreatitis, malignancy, snake bites, liver disease, transplant rejection, and transfusion reactions
---### Statement statement_0063 (Correct: True)
**True topic**: COPD Exacerbation (ID 21)

**Predicted topic**: Asthma Exacerbation (ID 14)

**Statement text**: maternal vitamin c supplementation at 500 mg daily during pregnancy can reduce wheezing incidence in offspring from 47% to 28% when mothers are exposed to tobacco smoke, while omega-3 polyunsaturated fatty acids in maternal diet reduce persistent wheeze development to 17% compared to 24% with omega-6 fatty acids.

#### Top 5 retrieved chunks
- **Rank 1**: topic Asthma Exacerbation (ID 14), source: data/cleaned_topics/Asthma Exacerbation/Pediatric Asthma.md, section: ## etiology
  - Passage: > the copenhagen prospective studies on asthma in childhood (copsac2010) reveals that 17% of children born to mothers with diets high in omega-3 polyunsaturated fatty acids developed persistent wheeze or operating room operating room asthma during the first 3 years of life compared to nearly 24% in the group with diets high in omega-6 polyunsaturated fatty acids. vitamins e and c and zinc may also have protective effects. administering vitamin c at a dose of 500 mg/d to pregnant mothers appears to offer protection against the harmful effects of tobacco exposure
- **Rank 2**: topic COPD Exacerbation (ID 21), source: data/cleaned_topics/COPD Exacerbation/Asthma and COPD Overlap.md, section: ## etiology
  - Passage: > the copenhagen prospective studies on asthma in childhood (copsac2010) reveals that 17% of children born to mothers with diets high in omega-3 polyunsaturated fatty acids developed persistent wheeze or operating room operating room asthma during the first 3 years of life compared to nearly 24% in the group with diets high in omega-6 polyunsaturated fatty acids. vitamins e and c and zinc may also have protective effects. administering vitamin c at a dose of 500 mg/d to pregnant mothers appears to offer protection against the harmful effects of tobacco exposure
- **Rank 3**: topic Asthma Exacerbation (ID 14), source: data/cleaned_topics/Asthma Exacerbation/Status Asthmaticus.md, section: ## etiology
  - Passage: > the copenhagen prospective studies on asthma in childhood (copsac2010) reveals that 17% of children born to mothers with diets high in omega-3 polyunsaturated fatty acids developed persistent wheeze or operating room operating room asthma during the first 3 years of life compared to nearly 24% in the group with diets high in omega-6 polyunsaturated fatty acids. vitamins e and c and zinc may also have protective effects. administering vitamin c at a dose of 500 mg/d to pregnant mothers appears to offer protection against the harmful effects of tobacco exposure
- **Rank 4**: topic Asthma Exacerbation (ID 14), source: data/cleaned_topics/Asthma Exacerbation/Pediatric Asthma.md, section: ## etiology
  - Passage: > . administering vitamin c at a dose of 500 mg/d to pregnant mothers appears to offer protection against the harmful effects of tobacco exposure. offspring of mothers who receive vitamin c supplementation exhibit a wheezing incidence of 28%, while those without vitamin c supplementation have a higher incidence of 47%.
- **Rank 5**: topic COPD Exacerbation (ID 21), source: data/cleaned_topics/COPD Exacerbation/Asthma and COPD Overlap.md, section: ## etiology
  - Passage: > . administering vitamin c at a dose of 500 mg/d to pregnant mothers appears to offer protection against the harmful effects of tobacco exposure. offspring of mothers who receive vitamin c supplementation exhibit a wheezing incidence of 28%, while those without vitamin c supplementation have a higher incidence of 47%.
---### Statement statement_0072 (Correct: True)
**True topic**: Pulmonary Hypertension (ID 65)

**Predicted topic**: Hypertensive Emergency (ID 40)

**Statement text**: mean pulmonary lung respiratory artery pressures in patients with high altitude pulmonary lung respiratory hypertension can normalize within 2 years of returning to lower altitudes, though pressures may increase again with return to high elevation.

#### Top 5 retrieved chunks
- **Rank 1**: topic Hypertensive Emergency (ID 40), source: data/cleaned_topics/Hypertensive Emergency/Malignant Hypertension.md, section: ## prognosis
  - Passage: > . if unsuccessful, various palliative procedures can aid in maximizing right ventricular cardiac output. few long-term studies have assessed the prognosis of haph. some research suggests that mean pulmonary artery pressures can normalize in as little as 2 years after descent to lower altitudes but can increase with a return to high elevation.
- **Rank 2**: topic Pulmonary Hypertension (ID 65), source: data/cleaned_topics/Pulmonary Hypertension/Pulmonary Hypertension.md, section: ## prognosis
  - Passage: > . if unsuccessful, various palliative procedures can aid in maximizing right ventricular cardiac output. few long-term studies have assessed the prognosis of haph. some research suggests that mean pulmonary artery pressures can normalize in as little as 2 years after descent to lower altitudes but can increase with a return to high elevation.
- **Rank 3**: topic Hypertensive Emergency (ID 40), source: data/cleaned_topics/Hypertensive Emergency/Malignant Hypertension.md, section: ## continuing education activity
  - Passage: > this activity describes the development of pulmonary hypertension due to high altitude. we describe a phenomenon known as high altitude pulmonary hypertension (haph), which is classically defined as a mean pulmonary arterial pressure greater than or operating room operating room equal to 25 mmhg on a right cardiac catheterization and results in pulmonary vascular remodeling. this condition can present in individuals who typically reside at altitudes greater than 2500 meters. the hypoxic stimulus of high altitude and individual genetic factors play a role in the pathophysiology of pulmonary vasoconstriction and, eventually, vascular remodeling that occurs with haph
- **Rank 4**: topic Pulmonary Hypertension (ID 65), source: data/cleaned_topics/Pulmonary Hypertension/Pulmonary Hypertension.md, section: ## continuing education activity
  - Passage: > this activity describes the development of pulmonary hypertension due to high altitude. we describe a phenomenon known as high altitude pulmonary hypertension (haph), which is classically defined as a mean pulmonary arterial pressure greater than or operating room operating room equal to 25 mmhg on a right cardiac catheterization and results in pulmonary vascular remodeling. this condition can present in individuals who typically reside at altitudes greater than 2500 meters. the hypoxic stimulus of high altitude and individual genetic factors play a role in the pathophysiology of pulmonary vasoconstriction and, eventually, vascular remodeling that occurs with haph
- **Rank 5**: topic Pulmonary Hypertension (ID 65), source: data/cleaned_topics/Pulmonary Hypertension/Pulmonary Hypertension.md, section: ## etiology
  - Passage: > ## etiology

at altitudes above 2500 meters, the barometric pressure is lower, which decreases the partial pressure of oxygen in the air. even though the fio2 remains at 21% at high altitudes, the amount of oxygen reaching the alveoli is much less than at sea level, resulting in hypoxia and hypoxemia. hypoxia triggers pulmonary vasoconstriction and increases pulmonary artery pressures throughout the lung respiratory. the belief is that vascular remodeling secondary to abnormal smooth muscle production, a decrease in the intrinsic availability of nitric oxide, and poorly understood genetic predisposition all contribute to the development of haph.
---### Statement statement_0098 (Correct: True)
**True topic**: Acute Liver Failure (ID 6)

**Predicted topic**: Multi-organ Failure (ID 50)

**Statement text**: adequate nutrition in acute hepatic liver failure patients should include protein administration at 1.0 to 1.5 grams per kilogram per day, while blood hematologic hemic glucose should be maintained between 160 to 200 mg/dl to prevent hypoglycemia from impaired glycogen production and gluconeogenesis.

#### Top 5 retrieved chunks
- **Rank 1**: topic Multi-organ Failure (ID 50), source: data/cleaned_topics/Multi-organ Failure/Multi-organ Failure.md, section: ## treatment / management
  - Passage: > 3. consider a fever workup including blood hematologic and urine cultures and start empirical antibiotics when required.

  4. monitor hepatic encephalopathy and protect airway (aspiration risk) should the patient show signs of worsening encephalopathy. these patients should be intubated and should be on a protocol to avoid cerebral edema.

  5. adequate nutrition with 1.0 to 1.5 gm of protein per kilogram per day should be administered.

  6. monitor for hypoglycemia and maintain blood hematologic glucose between 160 to 200.

  7. discontinue all home medications except the ones we identify essential to continue.

**specific treatment when the exact etiology is known**
- **Rank 2**: topic Acute Liver Failure (ID 6), source: data/cleaned_topics/Acute Liver Failure/Acute Liver Failure.md, section: ## treatment / management
  - Passage: > 3. consider a fever workup including blood hematologic and urine cultures and start empirical antibiotics when required.

  4. monitor hepatic encephalopathy and protect airway (aspiration risk) should the patient show signs of worsening encephalopathy. these patients should be intubated and should be on a protocol to avoid cerebral edema.

  5. adequate nutrition with 1.0 to 1.5 gm of protein per kilogram per day should be administered.

  6. monitor for hypoglycemia and maintain blood hematologic glucose between 160 to 200.

  7. discontinue all home medications except the ones we identify essential to continue.

**specific treatment when the exact etiology is known**
- **Rank 3**: topic Diabetic Ketoacidosis (ID 30), source: data/cleaned_topics/Diabetic Ketoacidosis/Adult Diabetic Ketoacidosis.md, section: ## pathophysiology
  - Passage: > ## pathophysiology

diabetes mellitus is characterized by insulin deficiency and increased plasma glucagon levels, which can be normalized by insulin replacement. normally, once serum glucose concentration increases, it enters pancreatic beta cells and leads to insulin production. insulin decreases hepatic liver glucose production by inhibiting glycogenolysis and gluconeogenesis. glucose uptake by skeletal muscle and adipose tissue is increased by insulin. both of these mechanisms result in the reduction of blood hematologic sugar. in diabetic ketoacidosis, insulin deficiency and increased counter-regulatory hormones can lead to increased gluconeogenesis, accelerated glycogenolysis, and impaired glucose utilization. this will ultimately cause worsening hyperglycemia.
- **Rank 4**: topic Meningitis (ID 48), source: data/cleaned_topics/Meningitis/Bacterial Meningitis.md, section: ## evaluation
  - Passage: > patients presumed to have bacterial meningitis should receive a lumbar puncture to obtain a cerebrospinal fluid (csf) sample. the csf should be sent for gram stain, culture, complete cell count (cbc complete blood hematologic hemic count complete blood hematologic hemic count), and glucose and protein levels. bacterial meningitis typically results in low glucose and high protein levels in the cerebrospinal fluid. as csf glucose levels are dependent on circulating serum glucose levels, the csf to serum glucose ratio is considered more reliable parameter for the diagnosis of acute bacterial meningitis than absolute csf glucose levels
- **Rank 5**: topic Acute Liver Failure (ID 6), source: data/cleaned_topics/Acute Liver Failure/Acute Liver Failure.md, section: ## treatment / management
  - Passage: > 3. metabolic disorders: hypoglycemia occurs due to impaired glycogen production and gluconeogenesis, and will need continuous infusions of 10% to 20% glucose. hypophosphatemia occurring due to atp consumption in the setting of hepatocyte necrosis requires aggressive repletion. alkalosis in alf is due to hyperventilation, and acidosis with a ph less than 7.3 portends 95% mortality in acetaminophen overdose if the patient does not undergo a liver transplant. hypoxemia may occur due to aspiration, acute respiratory distress syndrome, or operating room operating room pulmonary hemorrhage
---### Statement statement_0114 (Correct: True)
**True topic**: Acute Abdomen (ID 1)

**Predicted topic**: Respiratory Failure (ID 67)

**Statement text**: the classic presentation of acute cholangitis includes charcot's triad of right upper quadrant pain, fever, and jaundice, which occurs when a stone occludes the biliary or operating room operating room hepatic liver ducts leading to dilation and bacterial superinfection.

#### Top 5 retrieved chunks
- **Rank 1**: topic Respiratory Failure (ID 67), source: data/cleaned_topics/Respiratory Failure/Acute Respiratory Failure.md, section: ## history and physical
  - Passage: > acute cholangitis:**** acute cholangitis occurs when a stone occludes the biliary or operating room operating room hepatic ducts and presents with vague right upper quadrant pain, fever, and jaundice, known as charcot's triad. the obstruction results in dilation and bacterial superinfection of the duct.
- **Rank 2**: topic Acute Abdomen (ID 1), source: data/cleaned_topics/Acute Abdomen/Acute Abdomen.md, section: ## history and physical
  - Passage: > acute cholangitis:**** acute cholangitis occurs when a stone occludes the biliary or operating room operating room hepatic ducts and presents with vague right upper quadrant pain, fever, and jaundice, known as charcot's triad. the obstruction results in dilation and bacterial superinfection of the duct.
- **Rank 3**: topic Respiratory Failure (ID 67), source: data/cleaned_topics/Respiratory Failure/Acute Respiratory Failure.md, section: ## evaluation
  - Passage: > **right upper quadrant pain**

right upper quadrant pain is associated with the liver or operating room operating room biliary tree. however, as the liver only becomes painful when its capsule is stretched, the primary causes are related to the biliary tree. patients with right upper quadrant pain should undergo a thorough workup, including a complete blood count (cbc complete blood count complete blood count), serum electrolyte, aminotransferases, alkaline phosphatase, serum bilirubin, lipase, and amylase. in addition, an abdominal ultrasonography is the imaging modality of choice.

**epigastric pain**
- **Rank 4**: topic Acute Abdomen (ID 1), source: data/cleaned_topics/Acute Abdomen/Acute Abdomen.md, section: ## evaluation
  - Passage: > **right upper quadrant pain**

right upper quadrant pain is associated with the liver or operating room operating room biliary tree. however, as the liver only becomes painful when its capsule is stretched, the primary causes are related to the biliary tree. patients with right upper quadrant pain should undergo a thorough workup, including a complete blood count (cbc complete blood count complete blood count), serum electrolyte, aminotransferases, alkaline phosphatase, serum bilirubin, lipase, and amylase. in addition, an abdominal ultrasonography is the imaging modality of choice.

**epigastric pain**
- **Rank 5**: topic Abdominal Trauma (ID 0), source: data/cleaned_topics/Abdominal Trauma/Abdominal Gunshot Wounds.md, section: ## evaluation
  - Passage: > **components of the extended focused assessment with sonography for trauma (efast) ultrasonography**

1\. right upper quadrant (ruq) - when evaluating this region, the evaluator must consider that if the hemorrhage is present within the peritoneum, is most commonly found within the ruq. the anatomical structures found within a right upper quadrant include:

  * inferior pole of the right kidney nephric 

  * subphrenic space (between diaphragm and liver hepatic) 

  * hepatorenal space (between liver hepatic and kidney nephric, also known as morrison's pouch)

  * pleural space
---### Statement statement_0153 (Correct: True)
**True topic**: CT Angiogram (ID 90)

**Predicted topic**: Angiography (invasive) (ID 84)

**Statement text**: the ratio of contrast volume to calculated creatinine clearance should be kept below 2 to minimize contrast-induced nephropathy risk, with markedly increased cin risk when this ratio exceeds 3.

#### Top 5 retrieved chunks
- **Rank 1**: topic Angiography (invasive) (ID 84), source: data/cleaned_topics/Angiography (invasive)/Angiography.md, section: ## complications
  - Passage: > . however, this formula is less useful in high-risk patients such as those with anemia, diabetes, cardiac heart failure, and cardiogenic circulatory shock, and it is infrequently used in clinical settings. gurm et al. concluded that the ratio of contrast volume (cv) to the calculated creatinine clearance (ccc) of less than two is associated with a low incidence of cin. in contrast, the risk of cin is markedly increased when the ratio exceeds three.
- **Rank 2**: topic Angiography (invasive) (ID 84), source: data/cleaned_topics/Angiography (invasive)/Coronary Angiography.md, section: ## complications
  - Passage: > . however, this formula is less useful in high-risk patients such as those with anemia, diabetes, cardiac heart failure, and cardiogenic circulatory shock, and it is infrequently used in clinical settings. gurm et al. concluded that the ratio of contrast volume (cv) to the calculated creatinine clearance (ccc) of less than two is associated with a low incidence of cin. in contrast, the risk of cin is markedly increased when the ratio exceeds three.
- **Rank 3**: topic CT Angiogram (ID 90), source: data/cleaned_topics/CT Angiogram/CT Angiography of the Head and Neck.md, section: ## complications
  - Passage: > . however, this formula is less useful in high-risk patients such as those with anemia, diabetes, cardiac heart failure, and cardiogenic circulatory shock, and it is infrequently used in clinical settings. gurm et al. concluded that the ratio of contrast volume (cv) to the calculated creatinine clearance (ccc) of less than two is associated with a low incidence of cin. in contrast, the risk of cin is markedly increased when the ratio exceeds three.
- **Rank 4**: topic Angiography (invasive) (ID 84), source: data/cleaned_topics/Angiography (invasive)/Peripheral Angiography.md, section: ## complications
  - Passage: > . however, this formula is less useful in high-risk patients such as those with anemia, diabetes, cardiac heart failure, and cardiogenic circulatory shock, and it is infrequently used in clinical settings. gurm et al. concluded that the ratio of contrast volume (cv) to the calculated creatinine clearance (ccc) of less than two is associated with a low incidence of cin. in contrast, the risk of cin is markedly increased when the ratio exceeds three.
- **Rank 5**: topic CT Angiogram (ID 90), source: data/cleaned_topics/CT Angiogram/Computed Tomography Angiography.md, section: ## complications
  - Passage: > . however, this formula is less useful in high-risk patients such as those with anemia, diabetes, cardiac heart failure, and cardiogenic circulatory shock, and it is infrequently used in clinical settings. gurm et al. concluded that the ratio of contrast volume (cv) to the calculated creatinine clearance (ccc) of less than two is associated with a low incidence of cin. in contrast, the risk of cin is markedly increased when the ratio exceeds three.
---### Statement statement_0174 (Correct: True)
**True topic**: Acute Liver Failure (ID 6)

**Predicted topic**: Multi-organ Failure (ID 50)

**Statement text**: current survival rates for acute hepatic failure patients, including those undergoing liver transplantation, show 1-year survival greater than 65% overall, with recent registry data indicating up to 79% at 1 year and 72% at 5 years for transplant recipients.

#### Top 5 retrieved chunks
- **Rank 1**: topic Multi-organ Failure (ID 50), source: data/cleaned_topics/Multi-organ Failure/Multi-organ Failure.md, section: ## prognosis
  - Passage: > the expected clinical outcomes have drastically changed since alf was first defined approximately 50 years ago. the current 1-year survival rate of patients, including those undergoing liver transplantation, is greater than 65%. in the past, studies from the united states and europe had indicated a lower 1-year survival rate of patients with alf receiving a liver transplant when compared to their counterparts in patients with cirrhosis. however, the 2012 registry from the united states and europe indicates a higher survival rate up to 79% at 1 year and 72% at 5 years
- **Rank 2**: topic Acute Liver Failure (ID 6), source: data/cleaned_topics/Acute Liver Failure/Acute Liver Failure.md, section: ## prognosis
  - Passage: > the expected clinical outcomes have drastically changed since alf was first defined approximately 50 years ago. the current 1-year survival rate of patients, including those undergoing liver transplantation, is greater than 65%. in the past, studies from the united states and europe had indicated a lower 1-year survival rate of patients with alf receiving a liver transplant when compared to their counterparts in patients with cirrhosis. however, the 2012 registry from the united states and europe indicates a higher survival rate up to 79% at 1 year and 72% at 5 years
- **Rank 3**: topic Heart Failure (Acute_Chronic) (ID 38), source: data/cleaned_topics/Heart Failure (Acute_Chronic)/Congestive Heart Failure and Pulmonary Edema.md, section: ## prognosis
  - Passage: > the prognosis is worse for those with cardiac failure who are hospitalized; those with cardiac failure commonly require repeat hospitalizations and develop an intolerance for standard treatments as the disease progresses. data from the united states medicare beneficiaries hospitalized during 2006 showed 30-day and 1-year mortality rates postadmission of 10.8% and 30.7,% respectively. mortality outcomes at 1 year also demonstrate a clear relationship with age and increase from 22% for those aged 65 to 42.7% for patients aged 85 and older.
- **Rank 4**: topic Stroke (Ischemic_Hemorrhagic) (ID 75), source: data/cleaned_topics/Stroke (Ischemic_Hemorrhagic)/Acute Stroke.md, section: ## prognosis
  - Passage: > according to a large national cohort study conducted in australia and new zealand, among patients hospitalized with a first cerebrovascular accident (including ischemic cerebrovascular accident, ich, sah, and cases with unspecified type), the survival rates at 5 years and 10 years were 52.8% and 36.4%, respectively. the study also reported a cumulative incidence of cerebrovascular accident recurrence of 19.8% at 5 years and 26.8% at 10 years.
- **Rank 5**: topic Heart Failure (Acute_Chronic) (ID 38), source: data/cleaned_topics/Heart Failure (Acute_Chronic)/Congestive Heart Failure and Pulmonary Edema.md, section: ## prognosis
  - Passage: > the diagnosis of cardiac failure alone can be associated with a mortality rate greater than many cancers. despite advances made in cardiac failure treatments, the prognosis of the condition worsens over time, resulting in frequent hospital admissions and premature death. results from one recent study showed that patients recently diagnosed with new-onset cardiac failure had a mortality rate of 20.2% at 1 year and 52.6% at 5 years. the 1- and 5-year mortality rates also increase significantly based on the patient's age
---