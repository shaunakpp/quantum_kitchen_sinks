import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

from gate_quantum_kitchen_sinks import GateQuantumKitchenSink
from adiabatic_quantum_kitchen_sinks import AdiabaticQuantumKitchenSink


def linear_model(X_train, X_test, y_train, y_test):
    classifier = make_pipeline(StandardScaler(), SGDClassifier(max_iter=5000, tol=1e-4))
    classifier.fit(X_train, y_train)

    train_acc = classifier.score(X_train, y_train)
    test_acc = classifier.score(X_test, y_test)

    print(
        "Accuracy\n----- \n training: {}\n test:     {}"
          .format(train_acc, test_acc)
         )

    train_preds = classifier.predict(X_train)
    test_preds = classifier.predict(X_test)

    print(classification_report(y_test, test_preds))
    return train_preds, test_preds


digits = datasets.load_digits(n_class=2)

print('Gate Quantum Kitchen Sink Quantum Accuracy: ')
gqks = GateQuantumKitchenSink(10, 2)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
data = gqks.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.25, shuffle=False)

train_preds, test_preds = linear_model(X_train, X_test, y_train, y_test)


print('Adiabatic Quantum Kitchen Sink Accuracy: ')
aqks = AdiabaticQuantumKitchenSink(10, 2)
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
data = aqks.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.25, shuffle=False)

train_preds, test_preds = linear_model(X_train, X_test, y_train, y_test)
