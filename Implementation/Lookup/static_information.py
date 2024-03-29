uka_code_reason = {
    'J80' : 'ARDS',
    'J12' : 'Pneumonie',
    'J13' : 'Pneumonie',
    'J14' : 'Pneumonie',
    'J15' : 'Pneumonie',
    'J16' : 'Pneumonie',
    'J17' : 'Pneumonie',
    'J18' : 'Pneumonie',
    'J84' : 'Pneumonie',
    'T17' : 'Aspiration',
    'T75' : 'Beinahe-Ertrinken',
    'S20' : 'Thoraxtrauma',
    'S28' : 'Thoraxtrauma',
    'S29' : 'Thoraxtrauma',
    'A02' : 'Sepsis',
    'A20' : 'Sepsis',
    'A22' : 'Sepsis',
    'A26' : 'Sepsis',
    'A32' : 'Sepsis',
    'A39' : 'Sepsis',
    'A40' : 'Sepsis',
    'A41' : 'Sepsis',
    'A42' : 'Sepsis',
    'B37' : 'Sepsis',
    'R57' : 'Sepsis',
    'R65' : 'Sepsis',
    'K85' : 'Pankreatitis',
    'T79' : 'Fettembolie',
    'O88' : 'Fettembolie',
    'T80' : 'TRALI',
    'T20' : 'Verbrennungstrauma',
    'T21' : 'Verbrennungstrauma',
    'T22' : 'Verbrennungstrauma',
    'T23' : 'Verbrennungstrauma',
    'T24' : 'Verbrennungstrauma',
    'T25' : 'Verbrennungstrauma',
    'T26' : 'Verbrennungstrauma',
    'T27' : 'Verbrennungstrauma',
    'T28' : 'Verbrennungstrauma',
    'T29' : 'Verbrennungstrauma',
    'T30' : 'Verbrennungstrauma',
    'T31' : 'Verbrennungstrauma',
    'T32' : 'Verbrennungstrauma',
    'J81' : 'Lungenoedem',
    'I50' : 'Herzinsuffizienz',
    'R57' : 'Herzinsuffizienz',
    'I13' : 'Herzinsuffizienz',
    'I11' : 'Herzinsuffizienz',
    'E87' : 'Hypervolaemie'

}

other_code_reason = {
    'J80' : 'ARDS',
    'J12' : 'Pneumonie',
    'J13' : 'Pneumonie',
    'J14' : 'Pneumonie',
    'J15' : 'Pneumonie',
    'J16' : 'Pneumonie',
    'J17' : 'Pneumonie',
    'J18' : 'Pneumonie',
    'T17' : 'Aspiration',
    'T75.1' : 'Beinahe-Ertrinken',
    'W65' : 'Beinahe-Ertrinken',
    'W66' : 'Beinahe-Ertrinken',
    'W67' : 'Beinahe-Ertrinken',
    'W68' : 'Beinahe-Ertrinken',
    'W69' : 'Beinahe-Ertrinken',
    'W70' : 'Beinahe-Ertrinken',
    'S20.2' : 'Thoraxtrauma',
    'S28.0' : 'Thoraxtrauma',
    'S29.9' : 'Thoraxtrauma',
    'A02.1' : 'Sepsis',
    'A20.7' : 'Sepsis',
    'A22.7' : 'Sepsis',
    'A26.7' : 'Sepsis',
    'A32.7' : 'Sepsis',
    'A39.2' : 'Sepsis',
    'A39.3' : 'Sepsis',
    'A39.4' : 'Sepsis',
    'A40' : 'Sepsis',
    'A41' : 'Sepsis',
    'A42.7' : 'Sepsis',
    'B37.7' : 'Sepsis',
    'R57.2' : 'Sepsis',
    'R65.1' : 'Sepsis',
    'K85' : 'Pankreatitis',
    'O88.8' : 'Fettembolie',
    'T79.1' : 'Fettembolie',
    'T80.8' : 'TRALI',
    'T20' : 'Verbrennungstrauma',
    'T21' : 'Verbrennungstrauma',
    'T22' : 'Verbrennungstrauma',
    'T23' : 'Verbrennungstrauma',
    'T24' : 'Verbrennungstrauma',
    'T25' : 'Verbrennungstrauma',
    'T26' : 'Verbrennungstrauma',
    'T27' : 'Verbrennungstrauma',
    'T28' : 'Verbrennungstrauma',
    'T29' : 'Verbrennungstrauma',
    'T30' : 'Verbrennungstrauma',
    'T31' : 'Verbrennungstrauma',
    'T32' : 'Verbrennungstrauma',
    'J81' : 'Lungenoedem',
    'I50' : 'Herzinsuffizienz',
    'R57.0' : 'Herzinsuffizienz',
    'I13' : 'Herzinsuffizienz',
    'I11' : 'Herzinsuffizienz',
    'E87.7' : 'Hypervolaemie'
}

reason_index_dict = {
    'ARDS' : 0,
    'Pneumonie' : 1,
    'Aspiration' : 2,
    'Beinahe-Ertrinken' : 3,
    'Thoraxtrauma' : 4,
    'Sepsis' : 5,
    'Pankreatitis' : 6,
    'Fettembolie' : 7,
    'TRALI' : 8,
    'Verbrennungstrauma' : 9,
    'Herzinsuffizienz' : 10,
    'Lungenoedem' : 11,
    'Hypervolaemie' : 12
}

static_information_names = [
    'Alter',
    'Geschlecht',
    'Gewicht',
    'Groesse',
    'BMI',
    'Minimaler_Horowitz',
    'ARDS',
    'Pneumonie',
    'Aspiration', 
    'Beinahe-Ertrinken',
    'Thoraxtrauma',
    'Sepsis',
    'Pankreatitis',
    'Fettembolie',
    'TRALI',
    'Verbrennungstrauma',
    'Lungenoedem',
    'Herzinsuffizienz',
    'Hypervolaemie',
    'Zeitpunkt_minimaler_Horowitz',
    'admission_id'
]



static_dict = {
    'Alter' : 0,
    'Geschlecht' : 1,
    'Gewicht' : 2,
    'Groesse' : 3,
    'BMI' : 4,
    'Minimaler_Horowitz' : 5,
    'Zeitpunkt_minimaler_Horowitz' : 6,
    'admission_id' : 7,
    'ARDS' : 8,
    'Pneumonie' : 9,
    'Aspiration' : 10,  
    'Beinahe-Ertrinken' : 11,
    'Thoraxtrauma' : 12,
    'Sepsis' : 13,
    'Pankreatitis' : 14,
    'Fettembolie' : 15,
    'TRALI' : 16,
    'Verbrennungstrauma' : 17,
    'Lungenoedem' : 18,
    'Herzinsuffizienz' : 19,
    'Hypervolaemie' : 20

    
}

