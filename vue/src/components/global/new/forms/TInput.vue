<template>
  <input
    v-model="input"
    :class="['t-input', { 't-input--error': error }, { 't-input--small': small }]"
    :type="type || 'text'"
    :name="name || parse"
    :data-degree="degree"
    :autocomplete="'off'"
    @blur="change"
    @focus="focus"
  />
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
    label() {
      this.$el.focus()
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
    height: 24px;
    width: 109px;
    font-size: 12px;
    line-height: 24px;
  }
}
</style>
