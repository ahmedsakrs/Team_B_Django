from django.shortcuts import render
import joblib


def home(request):
    return render(request, 'home.html')


def result(request):
    model = joblib.load('model.sav')
    scaler = joblib.load('scaler.sav')
    features = [[request.GET['Term'], request.GET['Borrower APR'], request.GET['Estimated Return'],
                 request.GET['Prosper Rating'], request.GET['Prosper Score'], request.GET['Listing Category'],
                 request.GET['Employment Status Duration in Months'], request.GET['Is Borrower Home Owner'],
                 request.GET['Total Credit Lines Past 7 Years'], request.GET['Open Revolving Accounts'],
                 request.GET['Open Revolving Monthly Payment'], request.GET['Inquiries Last 6 Months'],
                 request.GET['Total Inquiries'], request.GET['Delinquencies Last 7 Years'],
                 request.GET['Revolving Credit Balance'], request.GET['Bankcard Utilization'],
                 request.GET['Available Bankcard Credit'], request.GET['Trades Never Delinquent'],
                 request.GET['Trades Opened Last 6 Months'], request.GET['Debt To Income Ratio'],
                 request.GET['Stated Monthly Income'], request.GET['Loan Months Since Origination'],
                 request.GET['Loan Original Amount'],
                 request.GET['LP Service Fees'], request.GET['Investors'], request.GET['Credit Score']]]

    features = scaler.transform(features)
    result = model.predict(features)

    return render(request, 'result.html', {'result': result})
