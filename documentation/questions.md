-   why is there so much extensive feature-engineering in the GBDT but not in the transformer model


-   why is there a fastformer model in kami? 

    the ebrec folder which contains this file is shared between the 2 folders kami and kfujikawa the file is not actually being used by kami




-   why is there an ensemble file code in kfujikawa's folder? kfujikawa\src\exp\v8xxx_ensemble\v8004_015_016_v1170_v1174.py

    This is an intermediate ensemble that combines predictions from both KAMI and
    KFujikawa pipelines, but it's not the final ensemble, and it's not part of the main process.

    Purpose:
    It Combines 2 KAMI models + 2 KFujikawa models with specific weights:
        - KAMI: 015_train_third (weight=2) + 016_catboost (weight=1)
        - KFujikawa: v1174 (weight=2) + v1170 (weight=1)

    Sugawarya reads individual model predictions directly, not ensemble outputs