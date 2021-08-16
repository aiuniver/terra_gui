<template>
  <div :class="['t-field', { 't-inline': inline }]">
    <label class="t-field__label" :for="parse"><slot>{{ label }}</slot></label>
    <input
      v-model="input"
      :class="['t-field__input', {small: small}]"
      :id="parse"
      :type="type"
      :name="name || parse"
      :value="value"
      @blur="change"
      :disabled="disabled"
      :data-degree="degree"
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
      type: [String, Number],
    },
    parse: String,
    name: String,
    inline: Boolean,
    disabled: Boolean,
    small: Boolean,
    degree: Number, // for serialize
  },
  data: () => ({
    isChange: false,
  }),
  computed: {
    input: {
      set(value) {
        this.$emit('input', value);
        this.isChange = true;
      },
      get() {
        return this.value;
      },
    },
  },
  methods: {
    change(e) {
      if (this.isChange) {
        let value = e.target.value;
        value = this.type === 'number' ? +value : value;
        this.$emit('change', { name: this.name, value });
        this.isChange = false;
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.t-field {
  // margin-bottom: 20px;
  &__label {
    text-align: left;
    color: #a7bed3;
    display: block;
    padding-bottom: 10px;
    line-height: 1.5;
    font-size: 0.75rem;
    text-overflow: ellipsis;
    white-space: nowrap;
    overflow: hidden;
  }
  &__input {
    color: #fff;
    border-color: #6c7883;
    background: #242f3d;
    height: 42px;
    padding: 0 10px;
    font-size: 0.875rem;
    font-weight: 400;
    border-radius: 4px;
    transition: border-color 0.3s ease-in-out, opacity 0.3s ease-in-out;
    &:focus {
      border-color: #fff;
    }
  }
  &__input.small{
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
  > label {
    width: 150px;
    max-width: 130px;
    padding: 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1.25;
    font-size: 0.75rem;
  }
  > input {
    height: 24px;
    font-size: 12px;
    line-height: 24px;
    width: 100px;
  }
}
</style>
