class Errors {
  constructor(callback = null) {
    this.error = {};
    this.cbFunc = callback;
  }

  has(field) {
    return Object.prototype.hasOwnProperty.call(this.error, field);
  }

  any() {
    return Object.keys(this.error).length > 0;
  }

  get(field) {
    if (this.error?.[field]) {
      return this.error[field];
    }
  }

  record(error) {
    if (typeof error === 'string') {
      this.error.message = error;
      if (typeof this.cbFunc === 'function') {
        this.cbFunc(error);
      }
    } else {
      this.error = error;
    }
  }

  clear(field = null) {
    if (field) {
      this.error[field] = null;
      delete this.error[field];
      return;
    }
    this.error = {};
  }
}

export default Errors;
