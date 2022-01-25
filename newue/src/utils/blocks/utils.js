const getOffsetRect = function (element) {
  let box = element.getBoundingClientRect();

  let scrollTop = window.pageYOffset;
  let scrollLeft = window.pageXOffset;

  let top = box.top + scrollTop;
  let left = box.left + scrollLeft;

  return { top: Math.round(top), left: Math.round(left) };
};

const mouseHelper = function (element, event) {
  let mouseX = event.pageX || event.clientX + document.documentElement.scrollLeft;
  let mouseY = event.pageY || event.clientY + document.documentElement.scrollTop;

  let offset = getOffsetRect(element);
  let x = mouseX - offset.left;
  let y = mouseY - offset.top;

  return {
    x: x,
    y: y,
  };
};

const debounce = function (func, wait, immediate) {
  let timeout;
  return function executedFunction () {
    const context = this;
    const args = arguments;
    const later = function () {
      timeout = null;
      if (!immediate) func.apply(context, args);
    };
    const callNow = immediate && !timeout;
    clearTimeout(timeout);
    timeout = setTimeout(later, wait);
    if (callNow) func.apply(context, args);
  };
};

export { mouseHelper, debounce };
