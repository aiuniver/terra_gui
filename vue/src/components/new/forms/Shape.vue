<template>
  <div :class="['t-field', { 't-inline': inline }]">
    <label class="t-field__label" @click="$el.getElementsByTagName('input')[0].focus()">{{ label }}</label>
    <input
      v-model="input"
      class="t-field__input"
      :type="type"
      :name="parse"
      :value="value"
      autocomplete="off"
      @blur="change"
      :disabled="disabled"
    />
  </div>
</template>

<script>
export default {
  name: 't-tuple-shape',
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
      type: [Array],
    },
    parse: String,
    name: String,
    inline: Boolean,
    disabled: Boolean,
  },
  data: () => ({
    isChange: false,
  }),
  computed: {
    input: {
      set(value) {
        this.$emit('input', value ? [value.split(',')] : []);
        this.isChange = true;
      },
      get() {
        return this.value.join();
      },
    },
  },
  methods: {
    change(e) {
      if (this.isChange) {
        let value = e.target.value;
        value = this.type === 'number' ? +value : value;

        this.$emit('change', { name: this.name, value: value ? [value.split(',')] : [] });
        this.isChange = false;
      }
    },
  },
};
</script>

<style lang="scss" scoped>
.t-field {
  margin-bottom: 10px;
  &__label {
    // width: 150px;
    // max-width: 130px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0 0 10px 0;
    line-height: 1.25;
    font-size: 0.75rem;
    text-overflow: ellipsis;
    white-space: nowrap;
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
    border: 1px solid #65B9F4;
    transition: border-color 0.3s ease-in-out, opacity 0.3s ease-in-out;
    &:focus {
      border-color: #fff;
    }
  }
}
.t-inline {
  display: flex;
  flex-direction: row-reverse;
  justify-content: flex-end;
  -webkit-box-pack: end;
  align-items: center;
  > label {
    width: auto;
    padding: 0 20px 0 10px;
    text-align: left;
    color: #a7bed3;
    display: block;
    margin: 0;
    line-height: 1.25;
    font-size: 0.75rem;
  }
  > input {
    height: 22px;
    font-size: 12px;
    line-height: 24px;
    width: 100px;
  }
}
</style>
