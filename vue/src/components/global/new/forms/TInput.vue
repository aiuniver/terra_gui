<template>
  <div :class="['t-input']">
    <input
      v-model="input"
      :class="['t-input__input', { 't-input__input--error': error }, { 't-input__input--small': small }]"
      :type="type || 'text'"
      :name="name || parse"
      :data-degree="degree"
      v-bind="$attrs"
      autocomplete="off"
      @blur="change"
      @focus="focus"
      :disabled="isDisabled"
      @mouseover="hover = true"
      @mouseleave="hover = false"
    />
    <div v-if="error && hover" :class="['t-field__hint', { 't-field__hint--big': !small }]">
      <span>{{ error }}</span>
    </div>
  </div>
</template>

<script>
export default {
  name: 't-input-new',
  props: {
    type: String,
    value: [String, Number],
    parse: String,
    name: String,
    small: Boolean,
    error: String,
    degree: Number, // for serialize
    disabled: [Boolean, Array],
  },
  data: () => ({
    isChange: false,
    hover: false,
  }),
  computed: {
    isDisabled() {
      if (Array.isArray(this.disabled)) {
        return !!this.disabled.includes(this.name);
      } else {
        return this.disabled;
      }
    },
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
    label() {
      this.$el.children[0].focus()
    },
    focus(e) {
      this.$emit('focus', e);
      if (this.error) {
        this.$emit('clean', true);
      }
    },
    change({ target }) {
      if (this.isChange) {
        const value = this.type === 'number' ? +target.value : target.value;
        this.$emit('change', { name: this.name, value });
        this.$emit('parse', { name: this.name, parse: this.parse, value });
        this.isChange = false;
      }
    },
  },
  created() {
    this.input = this.value;
  },
};
</script>

<style lang="scss" scoped>
.t-input {
  position: relative;
  &__input {
    width: 100%;
    height: 42px;
    color: #fff;
    background: #242f3d;
    padding: 0 10px;
    font-size: 0.875rem;
    font-weight: 400;
    border-radius: 4px;
    border: 1px solid #6c7883;
    transition: border-color 0.3s ease-in-out;
    &:focus {
      border-color: #fff;
    }
    &--small {
      width: 100%;
      height: 24px;
      font-size: 12px;
      line-height: 24px;
      padding: 0 5px;
    }
    &--error {
      border-color: #ca5035;
      color: #ca5035;
    }
  }
  .t-field__hint {
    user-select: none;
    position: absolute;
    height: 22px;
    display: flex;
    align-items: center;
    padding: 0 5px 0 5px;
    top: 25px;
    background-color: #ca5035;
    color: #fff;
    border-radius: 4px;
    z-index: 5;
    span {
      font-style: normal;
      font-weight: normal;
      font-size: 9px;
      line-height: 12px;
    }
    &--big {
      padding: 5px 25px;
      top: 100%;
      margin-top: 2px;
    }
  }
}
</style>
