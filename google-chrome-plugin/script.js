let isPerforming;
let isError;
let isSuccess;

function detectNews() {

    isPerforming = true;
    isError = false;
    isSuccess = false;
    updateControls();

    let xhr = new XMLHttpRequest();
    xhr.open('POST', 'http://paulrozhkin.ru:6257/api/fake/');
    xhr.setRequestHeader("Content-Type", "application/json;charset=UTF-8");

    const data = JSON.stringify({"news": $("#news-text").val()});
    xhr.send(data);

    xhr.onload = function () {
        // анализируем HTTP-статус ответа, если статус не 200, то произошла ошибка
        if (xhr.status === 200) {
            // alert(`Ошибка ${xhr.status}: ${xhr.statusText}`); // response -- это ответ сервера
            try {
                const jsonResponse = JSON.parse(xhr.response);

                const probability = Number(jsonResponse.probability).toFixed(2)

                if (jsonResponse.isReal) {
                    $("#news-text").val("This is real news with a probability " + probability + "%");
                } else {
                    $("#news-text").val("This is fake news with a probability " + probability + "%");
                }

                isSuccess = true;
            } catch (error) {
                isError = true;
            }
        } else { // если всё прошло гладко, выводим результат
            // alert(`Готово, получили ${xhr.response.length} байт`);
            isError = true;
        }

        isPerforming = false;
        updateControls();
    };

    xhr.onerror = function () {
        isPerforming = false;
        isError = true;
        updateControls();
    };
}

function updateControls() {
    newsUpdatedHandler();
    updateButton();
}

function updateButton() {

    const button = $('#detect-button');
    button.removeClass();
    button.addClass('btn')

    if (isError) {
        button.addClass('btn-danger')
    } else if (isSuccess) {
        button.addClass('btn-success')
    } else {
        button.addClass('btn-primary')
    }
}

function newsUpdatedHandler() {
    if ($('#news-text').val() === '' || isPerforming) {
        //Check to see if there is any text entered
        // If there is no text within the input ten disable the button
        $('#detect-button').prop('disabled', true);
    } else {
        //If there is text in the input, then enable the button
        $('#detect-button').prop('disabled', false);
    }
}

document.getElementById('detect-button').addEventListener('click', detectNews);
$('#news-text').keyup(newsUpdatedHandler)