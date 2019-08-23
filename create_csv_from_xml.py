import os
import xml.etree.ElementTree as ET
import random
import pandas as pd

# read the reports xml files and create the dataset tsv
reports_path = "IU-XRay/reports"

reports = os.listdir(reports_path)

reports.sort()

reports_with_no_image = []
reports_with_empty_sections = []
reports_with_no_impression = []
reports_with_no_findings = []

images_captions = {}
reports_with_images = {}
text_of_reports = {}

def get_new_csv_dictionary():
    return {'Image Index': [],
                  'Patient ID': [],
                  'Findings': [],
                  'Impression': [],
                  'Caption': []
                  }

all_data_csv_dictionary = get_new_csv_dictionary()
patient_id = 0
for report in reports:

    tree = ET.parse(os.path.join(reports_path, report))
    root = tree.getroot()
    img_ids = []
    # find the images of the report
    images = root.findall("parentImage")
    # if there aren't any ignore the report
    if len(images) == 0:
        reports_with_no_image.append(report)
    else:
        sections = root.find("MedlineCitation").find("Article").find("Abstract").findall("AbstractText")
        # find impression and findings sections
        for section in sections:
            if section.get("Label") == "FINDINGS":
                findings = section.text
            if section.get("Label") == "IMPRESSION":
                impression = section.text

        if impression is None and findings is None:
            reports_with_empty_sections.append(report)
        else:
            if impression is None:
                reports_with_no_impression.append(report)
                caption = findings
            elif findings is None:
                reports_with_no_findings.append(report)
                caption = impression
            else:
                caption = impression + " " + findings

            for image in images:
                images_captions[image.get("id") + ".png"] = caption
                img_ids.append(image.get("id") + ".png")
                all_data_csv_dictionary['Image Index'].append(image.get("id") + ".png")
                all_data_csv_dictionary['Patient ID'].append(patient_id)
                all_data_csv_dictionary['Findings'].append(findings)
                all_data_csv_dictionary['Impression'].append(impression)
                all_data_csv_dictionary['Caption'].append("startseq " + caption + " endseq")

            reports_with_images[report] = img_ids
            text_of_reports[report] = caption
            patient_id = patient_id + 1


def split_train_test():
    num_test_images=500
    num_of_images=len(all_data_csv_dictionary['Image Index'])
    test_indices=random.sample(range(0, num_of_images), num_test_images)

    test_csv_dictionary = get_new_csv_dictionary()
    train_csv_dictionary= get_new_csv_dictionary()

    def append_to_csv_dic(csv_dictionary,index):
        csv_dictionary['Image Index'].append(all_data_csv_dictionary['Image Index'][index])
        csv_dictionary['Patient ID'].append(all_data_csv_dictionary['Patient ID'][index])
        csv_dictionary['Findings'].append(all_data_csv_dictionary['Findings'][index])
        csv_dictionary['Impression'].append(all_data_csv_dictionary['Impression'][index])
        csv_dictionary['Caption'].append(all_data_csv_dictionary['Caption'][index])

    for i in range(num_of_images):
        if i in test_indices:
            append_to_csv_dic(test_csv_dictionary,i)
        else:
            append_to_csv_dic(train_csv_dictionary,i)
    return train_csv_dictionary, test_csv_dictionary

train_csv,test_csv=split_train_test()

def save_csv(csv_dictionary,csv_name):
    df = pd.DataFrame(csv_dictionary)
    df.to_csv(os.path.join("IU-XRay",csv_name), index=False)

save_csv(all_data_csv_dictionary,"all_data.csv")
save_csv(train_csv,"training_set.csv")
save_csv(test_csv,"testing_set.csv")

