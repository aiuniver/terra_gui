<template>
  <div class="toolbar">
    <ul class="toolbar__menu">
      <li
        v-for="({ title, disabled, icon }, index) of items"
        :key="'items_' + index"
        :class="['toolbar__menu--item']"
        :disabled="disabled"
      >
        <i :title="title" :class="['t-icon', icon, { active: active(index) }]" />
      </li>
    </ul>
  </div>
</template>

<script>
import { mapGetters } from 'vuex';
export default {
  name: 'Toolbar',
  props: {
    collabse: Array,
  },
  data: () => ({}),
  computed: {
    ...mapGetters({
      items: 'trainings/getToolbar',
    }),
  },
  methods: {
    active(index) {
      console.log(this.collabse)
      return this.collabse.includes(index.toString());
    },
    click() {
      // if (!this.items[index].disabled) {
      //   this.items[index].active = !active;
      // }
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
      &[disabled='disabled'] {
        opacity: 0.1;
        cursor: default;
      }
    }
  }
}
</style>