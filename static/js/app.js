document.addEventListener('DOMContentLoaded', function () {
  const form = document.getElementById('predict-form');
  const headlineInput = document.getElementById('headline');
  const clearBtn = document.getElementById('clear-btn');
  const result = document.getElementById('result');
  const resultLabel = document.getElementById('result-label');
  const resultHeadline = document.getElementById('result-headline');

  function showResult(label, headline) {
    resultLabel.textContent = label;
    resultLabel.classList.remove('real', 'fake');
    resultLabel.classList.add(label === 'FAKE' ? 'fake' : 'real');
    resultHeadline.textContent = headline;
    result.hidden = false;
    // small animation
    result.classList.remove('pop');
    void result.offsetWidth;
    result.classList.add('pop');
  }

  function showError(msg) {
    resultLabel.textContent = msg;
    resultLabel.classList.remove('real', 'fake');
    resultLabel.classList.add('error');
    resultHeadline.textContent = '';
    result.hidden = false;
  }

  form.addEventListener('submit', function (ev) {
    ev.preventDefault();
    const headline = headlineInput.value.trim();
    if (!headline) {
      showError('Please enter a headline');
      return;
    }

    resultLabel.textContent = '…';
    resultHeadline.textContent = '';
    result.hidden = false;

    fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ headline })
    })
      .then(r => r.json())
      .then(json => {
        if (json && json.label) {
          showResult(json.label, json.headline);
        } else if (json && json.error) {
          showError(json.error);
        } else {
          showError('Unexpected response');
        }
      })
      .catch(err => {
        console.error(err);
        showError('Network error');
      });
  });

  clearBtn.addEventListener('click', function () {
    headlineInput.value = '';
    result.hidden = true;
    headlineInput.focus();
  });

  // example chip clicks
  document.querySelectorAll('.chip').forEach(function (chip) {
    chip.addEventListener('click', function () {
      headlineInput.value = chip.textContent;
      headlineInput.focus();
    });
  });
});
