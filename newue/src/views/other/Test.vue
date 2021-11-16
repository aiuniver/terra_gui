<template>
  <main class="page-test">
    <scrollbar>
      <div class="cont">
        <div class="row">
          <div class="col">
            <h3>All svg</h3>
            <SvgContainer all />
          </div>
        </div>
        

      </div>
    </scrollbar> 
  </main>
</template>

<script>
export default {
  name: 'Test',
  components: {
      SvgContainer: () => import("@/components/app/SvgContainer"),
  },
  
};
</script>

<style lang="scss" scoped>
.page-test {
  width: 100%;
  height: 100%;
  overflow: hidden;
}
.cont {
  height: 100%;
  width: 100%;
  padding: 20px;
  .row {
    width: 100%;
    display: flex;
    margin: 40px 0 0 0;
  }
  .col {
    padding: 10px;
    height: 100%;
    flex: 1 1 33.3%;
  }
}
h3 {
  margin: 0 0 20px 0;
  text-decoration-line: underline;
}
</style>