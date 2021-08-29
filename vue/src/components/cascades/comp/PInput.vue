<template>
  <div class="field-form field-inline field-reverse">
    <slot>
      <label>{{ label }}</label>
    </slot>
    <input v-model="input" :type="type" />
  </div>
</template>

<script>
export default {
  name: 'PInput',
  props: {
    value: String,
    label: String,
    type: String
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
}
</script>

<style lang="scss" scoped>

</style>