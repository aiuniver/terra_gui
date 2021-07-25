<template>
  <div class="params">
    <scrollbar :style="height">
      <div class="params-container pa-5">
        <input type="text">
      </div>
    </scrollbar>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'
export default {
  name: "Settings",
  components: {
  },
  data: () => ({}),
  computed: {
  ...mapGetters({
    height: "settings/autoHeight",
  }),
  },
  methods: {
    click() {
      console.log();
    },
  },
};
</script>

<style lang="scss" scoped>
.params {
  flex-shrink: 0;
  width: 400px;
  border-left: #0e1621 solid 1px;
}
button {
  font-size: 0.875rem;
}
</style>