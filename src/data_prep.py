# the shared prep function + split
def general_prep_data(data ) -> preprocessed_data

def logreg_prep_data(data) -> processed for logreg
    preprocessed = general_prep_data(data )
    add other logreg features
    return preprocessed_logreg

def boosting_prep_data(data) -> processed for boost tree
    preprocessed = general_prep_data(data )
    add other boost tree features
    return preprocessed_boost

def split_data(preprocessed_data) -> statified split (80 /20)
