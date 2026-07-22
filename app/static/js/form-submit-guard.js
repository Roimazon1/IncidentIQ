(function () {
  "use strict";

  const submittingForms = new Map();
  const submitButtonSelector = [
    'button[type="submit"]',
    "button:not([type])",
    'input[type="submit"]',
    'input[type="image"]',
  ].join(", ");

  function preserveSubmitterValue(form, submitter) {
    if (!submitter || !submitter.name || submitter.disabled) {
      return null;
    }

    const submittedValue = document.createElement("input");
    submittedValue.type = "hidden";
    submittedValue.name = submitter.name;
    submittedValue.value = submitter.value;
    form.append(submittedValue);
    return submittedValue;
  }

  function showLoadingState(submitter) {
    if (!submitter) {
      return;
    }

    const loadingLabel = submitter.dataset.loadingLabel || "Submitting...";
    if (submitter instanceof HTMLInputElement) {
      submitter.value = loadingLabel;
      return;
    }

    submitter.replaceChildren();
    const spinner = document.createElement("span");
    spinner.className = "spinner-border spinner-border-sm me-2";
    spinner.setAttribute("aria-hidden", "true");
    submitter.append(spinner, document.createTextNode(loadingLabel));
  }

  function guardSubmission(event) {
    const form = event.target;
    if (!(form instanceof HTMLFormElement)) {
      return;
    }
    if (submittingForms.has(form)) {
      event.preventDefault();
      event.stopImmediatePropagation();
      return;
    }

    const submitter = event.submitter;
    const submittedValue = preserveSubmitterValue(form, submitter);
    const buttons = Array.from(form.querySelectorAll(submitButtonSelector));
    const buttonStates = buttons.map((button) => ({
      button,
      disabled: button.disabled,
      content: button instanceof HTMLInputElement ? button.value : button.innerHTML,
    }));

    submittingForms.set(form, { buttonStates, submittedValue });
    form.dataset.submitState = "submitting";
    form.setAttribute("aria-busy", "true");
    showLoadingState(submitter);
    buttons.forEach((button) => {
      button.disabled = true;
    });
  }

  function restoreForm(form, state) {
    state.buttonStates.forEach(({ button, disabled, content }) => {
      button.disabled = disabled;
      if (button instanceof HTMLInputElement) {
        button.value = content;
      } else {
        button.innerHTML = content;
      }
    });
    if (state.submittedValue) {
      state.submittedValue.remove();
    }
    delete form.dataset.submitState;
    form.removeAttribute("aria-busy");
  }

  document.addEventListener("submit", guardSubmission);
  window.addEventListener("pageshow", function () {
    submittingForms.forEach((state, form) => restoreForm(form, state));
    submittingForms.clear();
  });
})();
