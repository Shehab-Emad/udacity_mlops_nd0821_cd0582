# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

We use RandomForestClassifier Model from sklearn.model.RandomForestClassifier.

## Intended Use

Predict whether salary over $50K per year or not based on  cleaned census data

## Training Data

We have used 80% of the original dataset for the training purposes of the model.

## Evaluation Data

We have used 20% of the original dataset for evaluation purposes of the model.

## Metrics
We used three metrics are fbeta_score, precision_score and recall_score.

Precision:  0.8536
Recall:  0.7091
Fbeta: 0.7747

## Ethical Considerations

The model is not biased of people.

## Caveats and Recommendations

I recommend that data may increase to ensure that bias is minimized.
