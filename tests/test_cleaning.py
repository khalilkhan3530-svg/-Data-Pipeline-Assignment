import pandas as pd
from src.inspect_data import remove_duplicates


def test_remove_duplicates():
    data = pd.DataFrame({
        "Name": ["Ali", "Sara", "Ali"],
        "Marks": [80, 90, 80]
    })

    result = remove_duplicates(data)

    assert len(result) == 2
    assert result.duplicated().sum() == 0



from src.inspect_data import remove_missing_descriptions


def test_remove_missing_descriptions():
    data = pd.DataFrame({
        "Description": ["Book", None, "Pen"],
        "Quantity": [2, 3, 5]
    })

    result = remove_missing_descriptions(data)

    assert len(result) == 2
    assert result["Description"].isnull().sum() == 0    


from src.inspect_data import fill_missing_descriptions


def test_fill_missing_descriptions():
    data = pd.DataFrame({
        "StockCode": ["A1", "A1", "B2"],
        "Description": ["Book", None, "Pen"]
    })

    result = fill_missing_descriptions(data.copy())

    assert result["Description"].isnull().sum() == 0
    assert result.loc[1, "Description"] == "Book"    


from src.inspect_data import convert_date


def test_convert_date():
    data = pd.DataFrame({
        "InvoiceDate": ["12/1/2010 8:26", "12/1/2010 8:30"]
    })

    result = convert_date(data.copy())

    assert pd.api.types.is_datetime64_any_dtype(result["InvoiceDate"])



from src.inspect_data import validate_quantity


def test_validate_quantity():
    data = pd.DataFrame({
        "Quantity": [2, 5, 10]
    })

    result = validate_quantity(data)

    assert result == True


def test_validate_quantity_invalid():
    data = pd.DataFrame({
        "Quantity": [2, "five", 10]
    })

    result = validate_quantity(data)

    assert result == False    


from src.inspect_data import validate_missing_values


def test_validate_missing_values():
    data = pd.DataFrame({
        "Description": ["Book", "Pen", "Pencil"]
    })

    result = validate_missing_values(data)

    assert result == True    



def test_validate_missing_values_invalid():
    data = pd.DataFrame({
        "Description": ["Book", None, "Pencil"]
    })

    result = validate_missing_values(data)

    assert result == False



from src.inspect_data import validate_duplicates


def test_validate_duplicates():
    data = pd.DataFrame({
        "Name": ["Ali", "Sara", "Ahmed"],
        "Marks": [80, 90, 75]
    })

    result = validate_duplicates(data)

    assert result == True    



def test_validate_duplicates_invalid():
    data = pd.DataFrame({
        "Name": ["Ali", "Sara", "Ali"],
        "Marks": [80, 90, 80]
    })

    result = validate_duplicates(data)

    assert result == False    

