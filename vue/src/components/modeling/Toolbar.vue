<template>
  <div class="toolbar">
    <ul class="toolbar__menu">
      <template v-for="({ title, name, disabled, icon, hr }, i) of toolbar">
        <li class="toolbar__menu--item" :key="i" :disabled="disabled" @click.prevent="click(name)">
          <i :class="['icon', icon]" :title="title"></i>
        </li>
        <hr v-if="hr" :key="`hr_${i}`" />
      </template>
    </ul>
  </div>
</template>

<script>
export default {
  name: 'Toolbar',
  data: () => ({}),
  computed: {
    toolbar: {
      set(value) {
        this.$store.dispatch('modeling/setToolbar', value);
      },
      get() {
        return this.$store.getters['modeling/getToolbar'];
      },
    },
  },
  methods: {
    click(event) {
      this.$emit('actions', event);
    },
  },
};
</script>

<style lang="scss" scoped>
.toolbar {
  z-index: 10;
  width: 41px;
  flex-shrink: 0;
  position: relative;
  border-right: #0e1621 1px solid;
  &__menu {
    padding: 10px 0;
    list-style: none;
    &--item {
      width: 40px;
      height: 40px;
      width: 40px;
      height: 40px;
      display: flex;
      justify-content: center;
      align-items: center;
      cursor: pointer;
      &[disabled='disabled'] {
        opacity: 0.1;
        cursor: default;
      }
    }
  }
}

hr {
  border: none;
  color: #0e1621;
  background-color: #0e1621;
  height: 1px;
  margin: 10px 0px;
}
</style>