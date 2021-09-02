<template>
  <div :class="['t-field', { 't-inline': inline }]">
    <label class="t-field__label" @click="$el.getElementsByTagName('input')[0].focus()">
      <slot>{{ label }}</slot>
    </label>

    <input
      v-model="input"
      :class="['t-field__input', { small: small }, { 't-field__error': error }]"
      :type="type"
      :name="name || parse"
      :value="value"
      :disabled="disabled"
      :data-degree="degree"
      :autocomplete="'off'"
      @blur="change"
      @focus="focus"
    />
  </div>
</template>

<script>
export default {
  name: 't-input',
  props: {
    label: {
      type: String,
      default: 'Label',
    },
    type: {
      type: String,
      default: 'text',
    },
    value: {
      type: [String, Number, Array],
    },
    parse: String,
    name: String,
    inline: Boolean,
    disabled: Boolean,
    small: Boolean,
    error: String,
    degree: Number, // for serialize
  },
  data: () => ({
    isChange: false,
  }),
  computed: {
    input: {
      set(value) {
        // console.log(value)
        this.$emit('input', value);
        this.isChange = true;
      },
      get() {
        return this.value;
      },
    },
  },
  methods: {
    focus(e) {
      this.$emit('focus', e);
      if (this.error) {
        this.$emit('cleanError', true);
      }
    },
    change(e) {
      // if (+e.target.value > 99 && this.name === 'classes') e.target.value = '99'
      if (this.isChange) {
        let value = e.target.value;
        value = this.type === 'number' ? +value : value;
        this.$emit('change', { name: this.name, value });
        this.isChange = false;
      }
    },
  },
  created() {
    this.input = this.value
  }
};
</script>

<style lang="scss" scoped>
.t-field {
  margin-bottom: 10px;
  &__label {
    text-align: left;
    color: #a7bed3;
    display: block;
    padding-bottom: 10px;
    line-height: 1;
    font-size: 0.75rem;
    // text-overflow: ellipsis;
    // white-space: nowrap;
    overflow: hidden;
  }
  &__input {
    color: #fff;
    background: #242f3d;
    height: 42px;
    padding: 0 10px;
    font-size: 0.875rem;
    font-weight: 400;
    border-radius: 4px;
    border: 1px solid #6c7883;
    transition: border-color 0.3s ease-in-out, opacity 0.3s ease-in-out;
    &:focus {
      border-color: #fff;
    }
  }
  &__error {
    border-color: #b53b3b;
  }
  &__input.small {
    height: 24px;
  }
}
.t-inline {
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
  margin-bottom: 10px;
  align-items: center;
  .t-field__label {
    width: 150px;
    max-width: 130px;
    padding: 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1;
    font-size: 0.75rem;
  }
  .t-field__input {
    height: 24px;
    font-size: 12px;
    line-height: 24px;
    width: 180px;
  }
}
</style>
