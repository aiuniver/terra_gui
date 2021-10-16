import { debounce } from '@/utils/core/utils';
export default {
  props: {
    disabled: [Boolean, Array],
    name: String,
    small: Boolean,
    error: String,
    icon: String
  },
  computed: {
    isDisabled () {
      return Array.isArray(this.disabled) ? !!this.disabled.includes(this.name) : this.disabled;
    },
  },
  data: () => ({
    debounce: null,
  }),
  methods: {
    label () {
      this.$el.children[0].focus();
    },
  },
  created () {
    this.debounce = debounce(this.change, 300);
    if (this.$parent?.$options?._componentTag === 't-field') this.$parent.error = this.error;
  },
  watch: {
    error (value) {
      console.log(this.$parent?.$options?._componentTag);
      if (this.$parent?.$options?._componentTag === 't-field') this.$parent.error = value;
    },
  },
};
