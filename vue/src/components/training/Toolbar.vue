<template>
  <div class="toolbar project-training-toolbar">
    <div class="wrapper">
      <ul class="menu-section">
        <li
          v-for="({ title, active, disabled, icon }, index) of items"
          :key="'items_' + index"
          :class="{ active: active }"
          :disabled="disabled"
          @click="click(index, active)"
        >
          <span :title="title" :class="icon"></span>
        </li>
      </ul>
    </div>
  </div>
</template>

<script>
import { mapGetters } from "vuex";
export default {
  name: "Toolbar",
  data: () => ({}),
  computed: {
    ...mapGetters({
      items: "trainings/getToolbar",
    }),
  },
  methods: {
    click(index, active) {
      if (!this.items[index].disabled) {
        this.items[index].active = !active;
      }
    },
  },
};
</script>

<style scoped>
</style>