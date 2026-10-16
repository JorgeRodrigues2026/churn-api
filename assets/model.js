/* Modelo de churn exportado de models/churn_pipeline.joblib (Regressão Logística + pré-processamento).
   Gerado automaticamente — não edite à mão. */
window.CHURN_MODEL = {
 "model_name": "logistic_regression",
 "model_version": "1.0.0",
 "threshold": 0.5,
 "trained_at": "2026-10-15T18:51:20+00:00",
 "sklearn_version": "1.9.1",
 "intercept": -0.2668073707875404,
 "numeric": [
  {
   "name": "tenure",
   "median": 29.0,
   "mean": 32.34682286119986,
   "scale": 24.543913208049997,
   "coef": -1.2012675245809794
  },
  {
   "name": "MonthlyCharges",
   "median": 70.35,
   "mean": 64.81390663826767,
   "scale": 30.12860606513805,
   "coef": -0.4728739216950187
  },
  {
   "name": "TotalCharges",
   "median": 1390.725,
   "mean": 2285.535942492013,
   "scale": 2271.2149016772787,
   "coef": 0.560195033719919
  }
 ],
 "categorical": [
  {
   "name": "gender",
   "mode": "Male",
   "coefs": {
    "Female": -0.0983058176196241,
    "Male": -0.05936339345131077
   }
  },
  {
   "name": "SeniorCitizen",
   "mode": "No",
   "coefs": {
    "No": -0.1700288796129767,
    "Yes": 0.012359668542036812
   }
  },
  {
   "name": "Partner",
   "mode": "No",
   "coefs": {
    "No": -0.05311662772982959,
    "Yes": -0.10455258334112544
   }
  },
  {
   "name": "Dependents",
   "mode": "No",
   "coefs": {
    "No": 0.002653112887730242,
    "Yes": -0.16032232395867352
   }
  },
  {
   "name": "PhoneService",
   "mode": "Yes",
   "coefs": {
    "No": -0.027142167451002567,
    "Yes": -0.13052704361993103
   }
  },
  {
   "name": "MultipleLines",
   "mode": "No",
   "coefs": {
    "No": -0.20719561476996864,
    "No phone service": -0.027142167451002567,
    "Yes": 0.07666857115003213
   }
  },
  {
   "name": "InternetService",
   "mode": "Fiber optic",
   "coefs": {
    "DSL": -0.5323130169191899,
    "Fiber optic": 0.5966478671552355,
    "No": -0.22200406130700062
   }
  },
  {
   "name": "OnlineSecurity",
   "mode": "No",
   "coefs": {
    "No": 0.18590072836475457,
    "No internet service": -0.22200406130700062,
    "Yes": -0.12156587812871185
   }
  },
  {
   "name": "OnlineBackup",
   "mode": "No",
   "coefs": {
    "No": 0.07423376052745927,
    "No internet service": -0.22200406130700062,
    "Yes": -0.009898910291412379
   }
  },
  {
   "name": "DeviceProtection",
   "mode": "No",
   "coefs": {
    "No": 0.009042287244792406,
    "No internet service": -0.22200406130700062,
    "Yes": 0.0552925629912561
   }
  },
  {
   "name": "TechSupport",
   "mode": "No",
   "coefs": {
    "No": 0.18517033254753154,
    "No internet service": -0.22200406130700062,
    "Yes": -0.12083548231148335
   }
  },
  {
   "name": "StreamingTV",
   "mode": "No",
   "coefs": {
    "No": -0.12552242177738693,
    "No internet service": -0.22200406130700062,
    "Yes": 0.18985727201343353
   }
  },
  {
   "name": "StreamingMovies",
   "mode": "No",
   "coefs": {
    "No": -0.1358991914059435,
    "No internet service": -0.22200406130700062,
    "Yes": 0.20023404164199166
   }
  },
  {
   "name": "Contract",
   "mode": "Month-to-month",
   "coefs": {
    "Month-to-month": 0.7171487691161094,
    "One year": -0.028613993024859292,
    "Two year": -0.8462039871622061
   }
  },
  {
   "name": "PaperlessBilling",
   "mode": "Yes",
   "coefs": {
    "No": -0.27882924111062646,
    "Yes": 0.12116003003966326
   }
  },
  {
   "name": "PaymentMethod",
   "mode": "Electronic check",
   "coefs": {
    "Bank transfer (automatic)": -0.1847208871137264,
    "Credit card (automatic)": -0.16507065295774842,
    "Electronic check": 0.2671802782059152,
    "Mailed check": -0.07505794920540747
   }
  }
 ]
};
window.CHURN_METRICS = {
 "dataset": "IBM Telco Customer Churn",
 "n_samples": 7043,
 "split": {
  "train": 4225,
  "validation": 1409,
  "test": 1409
 },
 "selected_model": "logistic_regression",
 "model_version": "1.0.0",
 "validation_metrics": {
  "logistic_regression": {
   "accuracy": 0.7495,
   "precision": 0.5176,
   "recall": 0.8235,
   "f1_score": 0.6357,
   "roc_auc": 0.8594
  },
  "random_forest": {
   "accuracy": 0.7594,
   "precision": 0.5304,
   "recall": 0.8155,
   "f1_score": 0.6428,
   "roc_auc": 0.8585
  }
 },
 "test_metrics": {
  "accuracy": 0.7431,
  "precision": 0.5101,
  "recall": 0.8075,
  "f1_score": 0.6253,
  "roc_auc": 0.8319
 },
 "test_confusion_matrix": [
  [
   745,
   290
  ],
  [
   72,
   302
  ]
 ]
};
