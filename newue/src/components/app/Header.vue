<template>
  <div class="header">
    <div class="header__left">
      <!-- <Logo /> -->
    </div>
    <div class="app__center">
      <d-dropdown>
        <template v-slot:activator="{ on }">
          <d-button v-on="on" />
        </template>
        <ul class="list" @click="click">
          <li>1sdsdsd sdsdsdsd</li>
          <li>1sdsdsd sdsdsd</li>
          <li>1dsdsd sdsdsd</li>
        </ul>
      </d-dropdown>
    </div>
    <div class="header__right">

    </div>
  </div>
</template>

<script>
export default {
  name: 'app-header',
  components: {
    // Logo: () => import('@/components/app/comp/Logo'),
  },
  methods: {
    click(e) {
      console.log(e)
    }
  }
};
</script>

<style lang="scss" scoped>
.header {
  width: 100%;
  height: 52px;
  padding: 0 10px;
  display: flex;
  align-items: center;
}
</style>
