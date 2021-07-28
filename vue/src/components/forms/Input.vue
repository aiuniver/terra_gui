<template>
  <div class="field inline">
    <label class="field__label" :for="name">{{ label }}</label>
    <input
      class="field__input"
      :id="name"
      :type="type"
      :name="parse"
      :value="value"
    />
  </div>
</template>

<script>
export default {
  props: {
    label: {
      type: String,
      default: "Label",
    },
    type: {
      type: String,
      default: "text",
    },
    value: {
      type: [String, Number],
    },
    parse: String,
    name: String,
  },
};
</script>

<style lang="scss" scoped>
.inline {
  display: flex;
  .field__label {
    width: auto;
    padding: 0 20px 0 10px;
    text-align: left;
    width: 120px;
    margin: 0;
    padding: 0 10px 0 0;
    text-align: right;
    flex-direction: row-reverse;
    -webkit-box-pack: end;
    -moz-box-pack: end;
    -webkit-justify-content: flex-end;
    -ms-flex-pack: end;
    justify-content: flex-end;
  }
  .field__input {
    height: 22px;
    font-size: 0.75rem;
    max-width: 100px;
    width: 100px;
  }
}
.field {
  &__label {
    color: #a7bed3;
    display: block;
    margin: 0 0 10px 0;
    line-height: 1.25;
    font-size: 0.75rem;
  }
  &__input {
    // max-width: 100px !important;
  }
}
</style>
