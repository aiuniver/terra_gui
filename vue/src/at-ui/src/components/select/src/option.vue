<template>
  <li class="at-select__option"
    :class="[
      disabled ? 'at-select__option--disabled' : '',
      selected ? 'at-select__option--selected' : '',
      isFocus ? 'at-select__option--focus' : ''
    ]"
    @click.stop="doSelect"
    @mouseout.stop="blur"
    v-show="!hidden"
    ref="option"
  >
    <slot>{{ showLabel }}</slot>
  </li>
</template>

<script>
import Emitter from '@/at-ui/src/mixins/emitter'

export default {
  name: 'AtOption',
  mixins: [Emitter],
  inject: ['select'],
  props: {
    value: {
      type: [String, Number],
      required: true
    },
    label: {
      type: [String, Number]
    },
    disabled: {
      type: Boolean,
      default: false
    }
  },
  data () {
    return {
      selected: false,
      index: 0,
      isFocus: false,
      hidden: false,
      searchLabel: ''
    }
  },
  computed: {
    showLabel () {
      return this.label ? this.label : this.value
    }
  },
  methods: {
    doSelect () {
      if (this.disabled) return false
      this.dispatch('AtSelect', 'on-select-selected', this.value)
    },
    blur () {
      this.isFocus = false
    },
    queryChange (val) {
      const parsedQuery = val.replace(/(\^|\(|\)|\[|\]|\$|\*|\+|\.|\?|\\|\{|\}|\|)/g, '\\$1')
      this.hidden = !new RegExp(parsedQuery, 'i').test(this.searchLabel)
    }
  },
  mounted () {
    this.select.optionInstances.push(this)
    this.select.options.push({
      _instance: this,
      value: this.value,
      label: (typeof this.label === 'undefined') ? this.$el.innerHTML : this.label
    })
    this.searchLabel = this.$el.innerHTML
    this.$on('on-select-close', () => {
      this.isFocus = false
    })
    this.$on('on-query-change', val => {
      this.queryChange(val)
    })
  },
  beforeDestroy () {
    this.select.options.forEach((option, idx) => {
      if (option._instance === this) {
        this.select.onOptionDestroy(idx)
      }
    })
  }
}
</script>
