<template>
  <div :class="['t-field', { 't-inline': inline }]">
    <label class="t-field__label" :for="parse">{{ label }}</label>
    <input
      v-model="input"
      class="t-field__input"
      :id="parse"
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
    inline: Boolean,
  },
  computed: {
    input: {
      set(value) {
        this.$emit('input', value)
      },
      get() {
        return this.value
      }
    }
  }
};
</script>

<style lang="scss" scoped>
.t-field {
  margin-bottom: 20px;
  &__label {
    color: #a7bed3;
    display: block;
    margin: 0 0 10px 0;
    line-height: 1.25;
    font-size: 0.75rem;
    user-select: none;
  }
  &__input {
    color: #fff;
    border-color: #6C7883;
    background: #242F3D;
    height: 42px;
    padding: 0 10px;
    font-size: .875rem;
    font-weight: 400;
    border-radius: 4px;
    transition: border-color .3s ease-in-out, opacity .3s ease-in-out;
    &:focus{
      border-color: #fff;
    }
  }
}
.t-inline {
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
  margin-bottom: 10px;
  align-items: center;
  > label {
    width: auto;
    padding: 0 20px 0 10px;
    text-align: left;
    color: #A7BED3;
    display: block;
    margin: 0;
    line-height: 1.25;
    font-size: .75rem;
  }
  > input {
    height: 22px;
    font-size: 0.75rem;
    max-width: 100px;
    width: 100px;
  }
}
</style>
