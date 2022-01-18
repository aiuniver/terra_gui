<template>
  <div class="at-collapse__item"
    :class="{
      'at-collapse__item--active': isActive,
      'at-collapse__item--disabled': disabled,
      'at-collapse__item--not-change': notChange
    }">
    <div v-show="title" :class="['at-collapse__header', {'at-collapse__header--center': center }]" @click="toggle"> 
      <i v-if="!notChange" class="icon at-collapse__icon old__icon"></i>
      <slot name="title" v-if="$slots.title"></slot>
      <div v-else>{{ title }}</div>
    </div>
    <div class="at-collapse__body" v-show="isActive">
      <div class="at-collapse__content">
        <slot></slot>
      </div>
    </div>
  </div>
</template>

<script>
// import CollapseTransition from '@/at-ui/src/utils/collapse-transition'

export default {
  name: 'AtCollapseItem',
  components: {
    // CollapseTransition
  },
  props: {
    title: {
      type: String,
      default: ''
    },
    name: {
      type: String
    },
    disabled: {
      type: Boolean,
      default: false
    },
    notChange: Boolean,
    center: Boolean,
  },
  data () {
    return {
      index: 0,
      isActive: false
    }
  },
  methods: {
    toggle () {
      if (this.disabled) return false
      if (this.notChange) return;
      
      this.$parent.toggle({
        name: this.name || this.index,
        isActive: this.isActive
      })
    }
  }
}
</script>

